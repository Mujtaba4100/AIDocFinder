"""DocuFindAI Desktop (Local Search)

100% local, privacy-safe desktop app:
- User selects a folder
- Sync indexes supported files into local SQLite (text extraction + OCR + embeddings)
- Search runs locally over cached embeddings

Backend is NOT required for local search and is not modified.

PyInstaller notes:
- This app uses fastembed/onnxruntime when available.
- Model files may download on first run unless you prefetch and bundle them.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable so `from src...` works when running from src/ui/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Support running both from source tree and from a PyInstaller bundle.
if getattr(sys, 'frozen', False):
    # When frozen by PyInstaller, bundled files are extracted to _MEIPASS.
    bundle_root = getattr(sys, '_MEIPASS', ROOT)
    # If the bundle contains a top-level `src` folder, prefer that so imports like
    # `from src.ui.local_search...` resolve correctly.
    bundle_src = os.path.join(bundle_root, 'src')
    if os.path.isdir(bundle_src) and bundle_src not in sys.path:
        sys.path.insert(0, bundle_src)
    if bundle_root not in sys.path:
        sys.path.insert(0, bundle_root)
else:
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.utils import platform

from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.filemanager import MDFileManager

from src.ui.local_search.indexer import LocalIndexer
from src.ui.local_search.searcher import LocalSearcher


KV = r'''
<ResultRow>:
    text: root.row_title
    text_color: app.theme_cls.text_color
    md_bg_color: app.theme_cls.bg_normal
    size_hint_y: None
    height: dp(72)
    halign: 'left'
    on_release: app.open_result(root.full_path)

BoxLayout:
    orientation: 'vertical'
    padding: dp(12)
    spacing: dp(10)

    MDTopAppBar:
        title: 'DocuFindAI (Local)'
        elevation: 2

    MDCard:
        orientation: 'vertical'
        padding: dp(12)
        spacing: dp(10)
        size_hint_y: None
        height: dp(170)

        MDLabel:
            text: 'Folder'
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
                text: 'Browse Folder'
                on_release: app.open_folder_picker()

            MDRaisedButton:
                text: 'Sync Folder'
                disabled: (not app.folder_selected) or app.loading
                on_release: app.on_sync_clicked()

            MDRaisedButton:
                text: 'Clear Index'
                disabled: app.loading
                on_release: app.on_clear_index_clicked()

            MDSpinner:
                size_hint: None, None
                size: dp(32), dp(32)
                active: app.loading

        MDLabel:
            id: sync_status
            text: app.status_text
            theme_text_color: 'Hint'
            size_hint_y: None
            height: self.texture_size[1]

    MDCard:
        orientation: 'vertical'
        padding: dp(12)
        spacing: dp(10)
        size_hint_y: None
        height: dp(160)

        MDLabel:
            text: 'Search (local embeddings)'
            font_style: 'Subtitle1'
            size_hint_y: None
            height: self.texture_size[1]

        MDTextField:
            id: query_input
            hint_text: 'Enter search query'
            helper_text: 'Search is 100% local; files never leave your PC'
            helper_text_mode: 'persistent'
            on_text: app.on_query_changed(self.text)

        BoxLayout:
            size_hint_y: None
            height: dp(44)
            spacing: dp(8)

            MDTextField:
                id: limit_input
                hint_text: 'Limit'
                text: '10'
                input_filter: 'int'
                size_hint_x: 0.25

            MDRaisedButton:
                text: 'Search'
                disabled: (not app.folder_selected) or (not app.query_ready) or app.loading
                on_release: app.on_search_clicked()

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


from kivymd.uix.spinner import MDSpinner  # noqa: E402 (KV uses it)
from kivymd.uix.toolbar import MDTopAppBar  # noqa: E402 (KV uses it)
from kivymd.uix.card import MDCard  # noqa: E402 (KV uses it)
from kivymd.uix.textfield import MDTextField  # noqa: E402 (KV uses it)
from kivymd.uix.recycleview import MDRecycleView  # noqa: E402 (KV uses it)


class ResultRow(MDRaisedButton):
    row_title = StringProperty("")
    full_path = StringProperty("")


def _app_data_dir(app: MDApp) -> Path:
    return Path(app.user_data_dir)


def _copy_bundled_cache_if_present(dst_dir: Path) -> None:
    """Copy pre-fetched model cache from PyInstaller bundle to user_data_dir.

    If you prefetch into ./assets/fastembed_cache and bundle it as data, this will
    populate the user's cache directory on first run.
    """
    try:
        base = Path(getattr(sys, "_MEIPASS"))  # set by PyInstaller at runtime
    except Exception:
        return

    src = base / "fastembed_cache"
    if not src.exists() or not src.is_dir():
        return

    # Only copy if destination is empty/missing.
    if dst_dir.exists() and any(dst_dir.iterdir()):
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


class DocuFindLocalApp(MDApp):
    selected_folder = StringProperty("")
    folder_selected = BooleanProperty(False)
    query_ready = BooleanProperty(False)
    loading = BooleanProperty(False)
    status_text = StringProperty("Ready")

    def build(self):
        self.title = "DocuFindAI (Local)"
        self.theme_cls.primary_palette = "Blue"
        self.root = Builder.load_string(KV)

        self._dialog: Optional[MDDialog] = None
        self._file_manager: Optional[MDFileManager] = None

        # Local DB + model cache live under user_data_dir.
        data_dir = _app_data_dir(self)
        self.db_path = data_dir / "local_index.sqlite3"
        self.cache_dir = data_dir / "model_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # If user bundled a pre-fetched model cache, copy it now.
        _copy_bundled_cache_if_present(self.cache_dir)
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(self.cache_dir))

        self.indexer = LocalIndexer(db_path=self.db_path, cache_dir=self.cache_dir)
        self.searcher = LocalSearcher(db_path=self.db_path, cache_dir=self.cache_dir)

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
        self._dialog = MDDialog(
            title="DocuFindAI",
            text=text,
            buttons=[MDRaisedButton(text="OK", on_release=lambda *_: self._dialog.dismiss())],
        )
        self._dialog.open()

    def _toast(self, text: str) -> None:
        """Best-effort snackbar + status text."""
        self._set_status(text)
        try:
            from kivymd.uix.snackbar import Snackbar

            Snackbar(text=text).open()
        except Exception:
            # Non-fatal if Snackbar isn't available in this KivyMD build.
            pass

    def on_query_changed(self, text: str) -> None:
        self.query_ready = bool((text or "").strip())

    # ---------------------------
    # Clear index
    # ---------------------------
    def on_clear_index_clicked(self) -> None:
        if self.loading:
            return

        if self._dialog:
            try:
                self._dialog.dismiss()
            except Exception:
                pass

        def _cancel(*_args) -> None:
            if self._dialog:
                self._dialog.dismiss()

        def _confirm(*_args) -> None:
            if self._dialog:
                self._dialog.dismiss()
            self._start_clear_index()

        self._dialog = MDDialog(
            title="Clear local index?",
            text=(
                "This deletes the local search index database on this PC.\n\n"
                "You will need to click 'Sync Folder' again to rebuild it."
            ),
            buttons=[
                MDRaisedButton(text="Cancel", on_release=_cancel),
                MDRaisedButton(text="Clear", on_release=_confirm),
            ],
        )
        self._dialog.open()

    def _start_clear_index(self) -> None:
        self.loading = True
        self.root.ids.results_rv.data = []
        self._set_status("Clearing index...")
        thread = threading.Thread(target=self._background_clear_index, daemon=True)
        thread.start()

    def _background_clear_index(self) -> None:
        try:
            db_path = Path(self.db_path)
            deleted_any = False
            failed: list[str] = []

            for p in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
                try:
                    if p.exists():
                        p.unlink()
                        deleted_any = True
                except Exception:
                    if p.exists():
                        failed.append(str(p))

            if failed:
                raise RuntimeError("Could not delete index file(s): " + ", ".join(failed))

            # Recreate fresh index/search objects (they lazily create DB schema on use).
            self.indexer = LocalIndexer(db_path=self.db_path, cache_dir=self.cache_dir)
            self.searcher = LocalSearcher(db_path=self.db_path, cache_dir=self.cache_dir)

            msg = "Index cleared. Please Sync Folder again." if deleted_any else "Index already cleared."
            Clock.schedule_once(lambda dt, _m=msg: self._on_clear_index_done(_m), 0)
        except Exception as e:
            err = str(e)
            Clock.schedule_once(lambda dt, _e=err: self._on_clear_index_error(_e), 0)

    def _on_clear_index_done(self, msg: str) -> None:
        self.loading = False
        self.root.ids.results_rv.data = []
        self._toast(msg)

    def _on_clear_index_error(self, err: str) -> None:
        self.loading = False
        self._show_error(f"Clear index failed: {err}")

    # ---------------------------
    # Folder picker
    # ---------------------------
    def open_folder_picker(self) -> None:
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
                pass

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
        p = Path((folder or "").strip())
        if not p.exists() or not p.is_dir():
            self._show_error("Please choose a valid folder.")
            return
        self.selected_folder = str(p)
        self.folder_selected = True
        self._set_status("Ready")

    # ---------------------------
    # Sync / Index
    # ---------------------------
    def on_sync_clicked(self) -> None:
        if not self.folder_selected:
            self._show_error("Please select a folder first.")
            return
        # At least one embedder must be available.
        if not self.indexer.embedder.available and not self.indexer.clip.available:
            self._show_error(
                "No embedding backend is available.\n\n"
                f"Text embedder error: {self.indexer.embedder.init_error}\n"
                f"Image (CLIP) embedder error: {self.indexer.clip.init_error}\n\n"
                "Install fastembed + onnxruntime (recommended)."
            )
            return

        self.loading = True
        self.root.ids.results_rv.data = []
        self._set_status("Syncing folder...")
        thread = threading.Thread(target=self._background_sync, args=(self.selected_folder,), daemon=True)
        thread.start()

    def _background_sync(self, folder: str) -> None:
        try:
            def progress(msg: str) -> None:
                Clock.schedule_once(lambda dt, _m=msg: self._set_status(_m), 0)

            stats = self.indexer.index_folder(folder, progress=progress)
            final = f"Sync complete. Indexed: {stats.indexed}, skipped: {stats.skipped}, failed: {stats.failed}"
            Clock.schedule_once(lambda dt, _m=final: self._on_sync_done(_m), 0)
        except Exception as e:
            err = str(e)
            Clock.schedule_once(lambda dt, _e=err: self._on_sync_error(_e), 0)

    def _on_sync_done(self, msg: str) -> None:
        self.loading = False
        self._set_status(msg)

    def _on_sync_error(self, err: str) -> None:
        self.loading = False
        self._show_error(f"Sync failed: {err}")

    # ---------------------------
    # Local search
    # ---------------------------
    def on_search_clicked(self) -> None:
        if not self.folder_selected:
            self._show_error("Please select a folder first.")
            return
        query = (self.root.ids.query_input.text or "").strip()
        if not query:
            self._show_error("Please enter a query.")
            return
        if not self.searcher.embedder.available and not self.searcher.clip.available:
            self._show_error(
                "No embedding backend is available.\n\n"
                f"Text embedder error: {self.searcher.embedder.init_error}\n"
                f"Image (CLIP) embedder error: {self.searcher.clip.init_error}\n\n"
                "Install fastembed + onnxruntime (recommended)."
            )
            return

        try:
            limit = int(self.root.ids.limit_input.text or 10)
        except Exception:
            limit = 10
        limit = max(1, min(50, limit))

        self.loading = True
        self.root.ids.results_rv.data = []
        self._set_status("Searching...")
        thread = threading.Thread(target=self._background_search, args=(self.selected_folder, query, limit), daemon=True)
        thread.start()

    def _background_search(self, folder: str, query: str, limit: int) -> None:
        try:
            results = self.searcher.search(folder=folder, query=query, limit=limit)
            Clock.schedule_once(lambda dt, _r=results: self._on_search_done(_r), 0)
        except Exception as e:
            err = str(e)
            Clock.schedule_once(lambda dt, _e=err: self._on_search_error(_e), 0)

    def _on_search_done(self, results) -> None:
        self.loading = False
        if not results:
            self._set_status("No results (did you Sync Folder first?)")
            self.root.ids.results_rv.data = []
            return

        rv_data: List[Dict[str, Any]] = []
        for r in results:
            name = Path(r.rel_path).name
            title = f"[{getattr(r, 'source', 'local')}] {name}  (score: {r.score:.3f})\n{r.rel_path}"
            rv_data.append({"row_title": title, "full_path": r.path})
        self.root.ids.results_rv.data = rv_data
        self._set_status(f"Returned {len(rv_data)} results")

    def _on_search_error(self, err: str) -> None:
        self.loading = False
        self._show_error(f"Search failed: {err}")

    # ---------------------------
    # Open file
    # ---------------------------
    def open_result(self, full_path: str) -> None:
        if not full_path:
            return
        if platform == "android":
            self._set_status("Open file is desktop-only for now")
            return
        try:
            p = Path(full_path)
            if not p.exists():
                self._show_error("File not found on disk")
                return
            if platform == "win":
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif platform == "macosx":
                os.system(f'open "{p}"')
            else:
                os.system(f'xdg-open "{p}"')
        except Exception as e:
            self._show_error(f"Could not open file: {e}")


if __name__ == "__main__":
    DocuFindLocalApp().run()
