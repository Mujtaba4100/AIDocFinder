"""KivyMD UI for DocuFindAI (Desktop-first, Android-safe).

Backend (FastAPI) is assumed to be already running.

Critical behavior implemented here:
- Folder selection is REQUIRED before searching.
- Search requests are ALWAYS folder-scoped (send `path`to the backend).
- Recent folders (last 10) persist to a JSON file under `App.user_data_dir`.
- UI stays responsive: HTTP calls run in a background thread.

NOTE: This module intentionally does NOT import or initialize ML / VectorStore.
All searching is done via HTTP against the backend.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, StringProperty
from kivy.utils import platform

from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.menu import MDDropdownMenu

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


API_BASE = os.environ.get("DOCUFIND_API_BASE", "https://kstvx08j-8000.asse.devtunnels.ms/").rstrip("/")


KV = r'''
<ResultRow>:
    # Using a button as the row makes desktop clicks simple.
    text: root.row_title
    # Ensure result text is readable against theme background
    text_color: app.theme_cls.text_color
    md_bg_color: app.theme_cls.bg_normal
    size_hint_y: None
    height: dp(72)
    halign: "left"
    on_release: app.open_result(root.full_path)

BoxLayout:
    orientation: 'vertical'
    padding: dp(12)
    spacing: dp(10)

    MDTopAppBar:
        title: 'DocuFindAI'
        elevation: 2

    MDCard:
        orientation: 'vertical'
        padding: dp(12)
        spacing: dp(8)
        size_hint_y: None
        height: dp(140)

        MDLabel:
            text: 'Folder Selection'
            font_style: 'Subtitle1'
            size_hint_y: None
            height: self.texture_size[1]

        MDLabel:
            id: folder_label
            text: 'Selected Folder: ' + (app.selected_folder if app.selected_folder else 'None')
            theme_text_color: 'Hint'
            size_hint_y: None
            height: self.texture_size[1]

        BoxLayout:
            size_hint_y: None
            height: dp(44)
            spacing: dp(8)

            MDRaisedButton:
                id: browse_btn
                text: 'Browse Folder'
                on_release: app.open_folder_picker()

            MDRaisedButton:
                id: recent_btn
                text: 'Recent Folders'
                disabled: False if app.recent_paths else True
                on_release: app.open_recent_menu()

    MDCard:
        orientation: 'vertical'
        padding: dp(12)
        spacing: dp(10)
        size_hint_y: None
        height: dp(150)

        MDLabel:
            text: 'Search'
            font_style: 'Subtitle1'
            size_hint_y: None
            height: self.texture_size[1]

        MDTextField:
            id: query_input
            hint_text: 'Enter search query'
            helper_text: 'Search runs ONLY inside the selected folder'
            helper_text_mode: 'persistent'
            on_text: app.on_query_changed(self.text)

        BoxLayout:
            size_hint_y: None
            height: dp(44)
            spacing: dp(8)

            MDTextField:
                id: file_type_input
                hint_text: 'Type'
                text: app.file_type
                readonly: True
                on_focus: if self.focus: app.open_file_type_menu(self)
                size_hint_x: 0.35

            MDTextField:
                id: limit_input
                hint_text: 'Limit'
                text: '10'
                input_filter: 'int'
                size_hint_x: 0.25

            MDRaisedButton:
                id: search_btn
                text: 'Search'
                disabled: (not app.folder_selected) or (not app.query_ready) or app.loading
                on_release: app.on_search_clicked()

            MDSpinner:
                id: spinner
                size_hint: None, None
                size: dp(32), dp(32)
                active: app.loading

    MDLabel:
        id: status_label
        text: app.status_text
        theme_text_color: 'Hint'
        size_hint_y: None
        height: self.texture_size[1]

    MDLabel:
        text: 'Results'
        font_style: 'Subtitle1'
        size_hint_y: None
        height: self.texture_size[1]

    RecycleView:
        id: results_rv
        viewclass: 'ResultRow'
        bar_width: dp(8)
        scroll_type: ['bars', 'content']
        RecycleBoxLayout:
            default_size: None, dp(72)
            default_size_hint: 1, None
            size_hint_y: None
            height: self.minimum_height
            orientation: 'vertical'
            spacing: dp(8)
'''


class ResultRow(MDRaisedButton):
    row_title = StringProperty("")
    full_path = StringProperty("")


class DocuFindApp(MDApp):
    """Desktop-first KivyMD client for DocuFindAI backend."""

    selected_folder = StringProperty("")
    folder_selected = BooleanProperty(False)
    query_ready = BooleanProperty(False)
    loading = BooleanProperty(False)
    status_text = StringProperty("")

    file_type = StringProperty("all")
    file_type_options = ["all", "document", "image"]

    recent_paths = ListProperty([])
    results = ListProperty([])

    def build(self):
        self.title = "DocuFindAI"
        self.theme_cls.primary_palette = "Blue"
        self.root = Builder.load_string(KV)

        self._dialog: Optional[MDDialog] = None
        self._file_type_menu: Optional[MDDropdownMenu] = None
        self._recent_menu: Optional[MDDropdownMenu] = None

        self.recent_paths = self._load_recent_paths()
        self._init_file_type_menu()
        self._refresh_recent_menu()
        self._set_status(f"Backend: {API_BASE}")

        # Native folder picker on desktop; Android-safe fallback will be shown if needed.
        self._file_manager: Optional[MDFileManager] = None
        return self.root

    # ---------------------------
    # UI helpers
    # ---------------------------
    def _set_status(self, text: str) -> None:
        self.status_text = text or ""

    def _show_error(self, text: str) -> None:
        if self._dialog:
            try:
                self._dialog.dismiss()
            except Exception:
                pass
        self._dialog = MDDialog(title="DocuFindAI", text=text, buttons=[MDRaisedButton(text="OK", on_release=lambda *_: self._dialog.dismiss())])
        self._dialog.open()

    def on_query_changed(self, text: str) -> None:
        self.query_ready = bool((text or "").strip())

    def _init_file_type_menu(self) -> None:
        items = []
        for t in self.file_type_options:
            items.append(
                {
                    "text": t,
                    "height": dp(48),
                    "on_release": lambda x=t: self._set_file_type(x),
                }
            )
        self._file_type_menu = MDDropdownMenu(items=items, width_mult=3)

    def _set_file_type(self, value: str) -> None:
        self.file_type = value
        try:
            self.root.ids.file_type_input.text = value
        except Exception:
            pass
        if self._file_type_menu:
            self._file_type_menu.dismiss()

    def open_file_type_menu(self, caller) -> None:
        if not self._file_type_menu:
            self._init_file_type_menu()
        self._file_type_menu.caller = caller
        self._file_type_menu.open()

    # ---------------------------
    # Folder selection
    # ---------------------------
    def open_folder_picker(self) -> None:
        """Open a folder picker.

        Desktop (Windows/Linux/macOS): native folder chooser via tkinter.
        Android-safe fallback: use KivyMD file manager in directory mode.
        """
        if platform in ("win", "linux", "macosx"):
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                folder = filedialog.askdirectory()
                try:
                    root.destroy()
                except Exception:
                    pass
                if folder:
                    self.set_selected_folder(folder)
                return
            except Exception:
                # Fall through to file-manager-based selection.
                pass

        # Android-safe fallback (directory browsing)
        if not self._file_manager:
            self._file_manager = MDFileManager(
                exit_manager=self._close_file_manager,
                select_path=self._select_path_from_file_manager,
                preview=False,
                selector="folder",
            )
        start_path = self.selected_folder or str(Path.home())
        self._file_manager.show(start_path)

    def _close_file_manager(self, *args) -> None:
        if self._file_manager:
            self._file_manager.close()

    def _select_path_from_file_manager(self, path: str) -> None:
        self._close_file_manager()
        self.set_selected_folder(path)

    def set_selected_folder(self, folder: str) -> None:
        folder = (folder or "").strip()
        if not folder:
            return
        p = Path(folder)
        if not p.exists() or not p.is_dir():
            self._show_error("Please choose a valid folder.")
            return

        self.selected_folder = str(p)
        self.folder_selected = True
        self._save_recent_path(self.selected_folder)
        self._refresh_recent_menu()
        self._set_status("Ready")

    # ---------------------------
    # Recent folders
    # ---------------------------
    def _recent_paths_file(self) -> Path:
        return Path(self.user_data_dir) / "recent_paths.json"

    def _load_recent_paths(self) -> List[str]:
        try:
            fp = self._recent_paths_file()
            if fp.exists():
                data = json.loads(fp.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    # Keep only strings.
                    return [str(x) for x in data if isinstance(x, str)]
        except Exception:
            pass
        return []

    def _save_recent_path(self, path: str) -> None:
        if not path:
            return
        # Normalize + de-duplicate
        path = str(Path(path))
        current = [p for p in self.recent_paths if p != path]
        current.insert(0, path)
        self.recent_paths = current[:10]
        try:
            fp = self._recent_paths_file()
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps(self.recent_paths, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # Non-fatal
            pass

    def _refresh_recent_menu(self) -> None:
        items = []
        for p in self.recent_paths:
            items.append(
                {
                    "text": p,
                    "height": dp(48),
                    "on_release": lambda x=p: self._select_recent_folder(x),
                }
            )
        self._recent_menu = MDDropdownMenu(items=items, width_mult=6)

    def open_recent_menu(self) -> None:
        if not self.recent_paths:
            self._show_error("No recent folders yet. Use 'Browse Folder' first.")
            return
        self._refresh_recent_menu()
        self._recent_menu.caller = self.root.ids.recent_btn
        self._recent_menu.open()

    def _select_recent_folder(self, folder: str) -> None:
        if self._recent_menu:
            self._recent_menu.dismiss()
        self.set_selected_folder(folder)

    # ---------------------------
    # Search (folder-scoped only)
    # ---------------------------
    def on_search_clicked(self) -> None:
        query = (self.root.ids.query_input.text or "").strip()
        if not self.folder_selected:
            self._show_error("Please select a folder first.")
            return
        if not query:
            self._show_error("Please enter a search query.")
            return
        if requests is None:
            self._show_error("Python package 'requests' is not available. Install dependencies and try again.")
            return

        try:
            limit = int(self.root.ids.limit_input.text or 10)
        except Exception:
            limit = 10
        limit = max(1, min(50, limit))

        self.loading = True
        self.results = []
        self.root.ids.results_rv.data = []
        self._set_status("Searching...")

        thread = threading.Thread(
            target=self._background_search,
            args=(query, self.selected_folder, self.file_type, limit),
            daemon=True,
        )
        thread.start()

    def _background_search(self, query: str, folder: str, file_type: str, limit: int) -> None:
        try:
            payload = {
                "query": query,
                "file_type": file_type,
                "limit": limit,
                "path": folder,
            }
            resp = requests.post(f"{API_BASE}/search/", json=payload, timeout=180)
            if resp.status_code != 200:
                data: Dict[str, Any] = {"error": f"Server returned {resp.status_code}: {resp.text}"}
            else:
                data = resp.json()
            Clock.schedule_once(lambda dt: self._display_results(data), 0)
        except Exception as e:
            err = str(e)
            Clock.schedule_once(lambda dt, _err=err: self._display_results({"error": _err}), 0)

    def _display_results(self, results: Any) -> None:
        self.loading = False

        if results is None:
            self._set_status("No results returned.")
            return

        if isinstance(results, dict) and results.get("error"):
            self._set_status(f"Error: {results.get('error')}")
            return

        items: List[Dict[str, Any]] = []
        if isinstance(results, dict) and isinstance(results.get("results"), list):
            items = results["results"]
        elif isinstance(results, list):
            items = results
        else:
            self._set_status("Unexpected response from backend.")
            return

        if not items:
            self._set_status("No results found in selected folder.")
            self.root.ids.results_rv.data = []
            return

        folder = Path(self.selected_folder)
        rv_data = []
        for it in items:
            name = str(it.get("file") or it.get("id") or "unknown")
            score = it.get("score")
            try:
                score_val = float(score) if score is not None else 0.0
            except Exception:
                score_val = 0.0

            # Backend folder-search returns file names; derive full path inside selected folder.
            candidate = Path(name)
            full_path = str(candidate) if candidate.is_absolute() else str(folder / name)
            rel = full_path
            try:
                rel = str(Path(full_path).relative_to(folder))
            except Exception:
                pass

            title = f"{name}  (score: {score_val:.3f})"
            rv_data.append({"row_title": title, "full_path": full_path, "text": rel})

        self.root.ids.results_rv.data = rv_data
        self._set_status(f"Returned {len(rv_data)} results (folder-scoped).")

    # ---------------------------
    # Result click action
    # ---------------------------
    def open_result(self, full_path: str) -> None:
        """Desktop-only: open a result file with the OS default handler."""
        if not full_path:
            return
        if platform == "android":
            # Android support comes later; avoid breaking the app now.
            self._set_status("Open file is desktop-only for now.")
            return

        try:
            p = Path(full_path)
            if not p.exists():
                self._show_error("File not found on disk. The backend returned a name that does not exist locally.")
                return
            if platform == "win":
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif platform == "macosx":
                os.system(f'open "{p}"')
            else:
                os.system(f'xdg-open "{p}"')
        except Exception as e:
            self._show_error(f"Could not open file: {e}")


if __name__ == '__main__':
    DocuFindApp().run()
