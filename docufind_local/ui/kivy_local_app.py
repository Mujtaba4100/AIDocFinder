"""DocuFindLocal Desktop (Local Search) - Responsive UI

100% local, privacy-safe desktop app with responsive design.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty, StringProperty
from kivy.utils import platform

from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton, MDFlatButton, MDIconButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.filemanager import MDFileManager

# When running this file directly (python path/to/kivy_local_app.py),
# ensure the repository root is on sys.path so the `docufind_local`
# package can be imported. Running as a module (python -m ...) does
# not need this.
if __name__ == "__main__" and __package__ is None:
    from pathlib import Path
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from docufind_local.local_search.indexer import LocalIndexer
from docufind_local.local_search.searcher import LocalSearcher


KV = r'''
#:import Window kivy.core.window.Window

<ResultRow>:
    orientation: 'vertical'
    size_hint_y: None
    height: dp(120)
    padding: dp(12), dp(8)
    spacing: dp(4)
    md_bg_color: app.theme_cls.bg_light
    radius: [dp(8)]
    elevation: 1
    
    MDLabel:
        text: root.file_name
        font_style: 'Subtitle1'
        bold: True
        size_hint_y: None
        height: self.texture_size[1]
        text_size: self.width, None
        
    MDLabel:
        text: root.file_path
        font_style: 'Caption'
        theme_text_color: 'Secondary'
        size_hint_y: None
        height: self.texture_size[1]
        text_size: self.width, None
        
    MDLabel:
        text: root.score_text
        font_style: 'Caption'
        theme_text_color: 'Hint'
        size_hint_y: None
        height: self.texture_size[1]

    BoxLayout:
        size_hint_y: None
        height: dp(40)
        spacing: dp(8)
        padding: dp(4), 0
        
        MDRaisedButton:
            text: 'Open File'
            icon: 'file-document'
            on_release: app.open_result(root.full_path)
            
        MDRaisedButton:
            text: 'Show in Folder'
            icon: 'folder-open'
            on_release: app.open_file_location(root.full_path)

<ExpandableCard@MDCard>:
    orientation: 'vertical'
    padding: dp(16)
    spacing: dp(12)
    size_hint_y: None
    radius: [dp(12)]
    elevation: 2
    md_bg_color: app.theme_cls.bg_normal

ScrollView:
    do_scroll_x: False
    do_scroll_y: True
    bar_width: dp(10)
    scroll_type: ['bars', 'content']
    
    BoxLayout:
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(16)
        size_hint_y: None
        height: self.minimum_height

        MDTopAppBar:
            title: 'DocuFindLocal - Privacy-First Search'
            elevation: 4
            size_hint_y: None
            height: dp(56)
            md_bg_color: app.theme_cls.primary_color
            left_action_items: [['folder-search', lambda x: None]]

        # Folder Selection Card
        ExpandableCard:
            height: folder_box.height + dp(32)
            
            BoxLayout:
                id: folder_box
                orientation: 'vertical'
                spacing: dp(12)
                size_hint_y: None
                height: self.minimum_height

                MDLabel:
                    text: '📁 Folder Selection'
                    font_style: 'H6'
                    bold: True
                    size_hint_y: None
                    height: self.texture_size[1]

                MDCard:
                    orientation: 'vertical'
                    padding: dp(12)
                    size_hint_y: None
                    height: dp(60)
                    md_bg_color: app.theme_cls.bg_dark if app.folder_selected else app.theme_cls.bg_light
                    radius: [dp(8)]
                    
                    MDLabel:
                        text: app.selected_folder if app.selected_folder else 'No folder selected'
                        theme_text_color: 'Primary' if app.folder_selected else 'Hint'
                        size_hint_y: None
                        height: self.texture_size[1]
                        text_size: self.width, None
                        halign: 'left'

                BoxLayout:
                    size_hint_y: None
                    height: dp(48)
                    spacing: dp(8)

                    MDRaisedButton:
                        text: 'Browse'
                        icon: 'folder-open'
                        on_release: app.open_folder_picker()
                        md_bg_color: app.theme_cls.primary_color

                    MDRaisedButton:
                        text: 'Sync'
                        icon: 'sync'
                        disabled: (not app.folder_selected) or app.loading
                        on_release: app.on_sync_clicked()
                        md_bg_color: app.theme_cls.accent_color if not self.disabled else app.theme_cls.disabled_hint_text_color

                    MDFlatButton:
                        text: 'Clear'
                        disabled: app.loading
                        on_release: app.on_clear_index_clicked()

                BoxLayout:
                    size_hint_y: None
                    height: dp(32) if app.loading else 0
                    opacity: 1 if app.loading else 0
                    spacing: dp(8)
                    
                    MDSpinner:
                        size_hint: None, None
                        size: dp(24), dp(24)
                        active: app.loading
                        
                    MDLabel:
                        text: app.status_text
                        theme_text_color: 'Primary'
                        valign: 'center'

                MDLabel:
                    text: app.status_text
                    theme_text_color: 'Hint'
                    size_hint_y: None
                    height: self.texture_size[1] if not app.loading else 0
                    opacity: 0 if app.loading else 1

        # File Type Filter Card
        ExpandableCard:
            height: filetype_box.height + dp(32)
            
            BoxLayout:
                id: filetype_box
                orientation: 'vertical'
                spacing: dp(12)
                size_hint_y: None
                height: self.minimum_height

                MDLabel:
                    text: '🗂️ File Types'
                    font_style: 'H6'
                    bold: True
                    size_hint_y: None
                    height: self.texture_size[1]

                GridLayout:
                    cols: 2 if Window.width < dp(600) else 4
                    spacing: dp(8)
                    size_hint_y: None
                    height: self.minimum_height
                    row_default_height: dp(44)

                    MDRaisedButton:
                        text: 'All Files'
                        on_release: app.set_file_type('all')
                        md_bg_color: app.theme_cls.primary_color if app.selected_file_type == 'all' else app.theme_cls.bg_dark

                    MDRaisedButton:
                        text: 'Text'
                        on_release: app.set_file_type('text')
                        md_bg_color: app.theme_cls.primary_color if app.selected_file_type == 'text' else app.theme_cls.bg_dark

                    MDRaisedButton:
                        text: 'Documents'
                        on_release: app.set_file_type('doc')
                        md_bg_color: app.theme_cls.primary_color if app.selected_file_type == 'doc' else app.theme_cls.bg_dark

                    MDRaisedButton:
                        text: 'Images'
                        on_release: app.set_file_type('image')
                        md_bg_color: app.theme_cls.primary_color if app.selected_file_type == 'image' else app.theme_cls.bg_dark

                MDLabel:
                    text: app.file_type_extensions
                    theme_text_color: 'Hint'
                    font_style: 'Caption'
                    size_hint_y: None
                    height: self.texture_size[1]
                    text_size: self.width, None

                BoxLayout:
                    orientation: 'horizontal'
                    size_hint_y: None
                    height: dp(40)
                    spacing: dp(8)
                    
                    MDCheckbox:
                        id: recursive_checkbox
                        active: app.search_recursive
                        on_active: app.set_search_recursive(self.active)
                        size_hint: None, None
                        size: dp(32), dp(32)
                        
                    MDLabel:
                        text: "Search subfolders recursively"
                        valign: 'center'
                        theme_text_color: "Primary"

        # Search Card
        ExpandableCard:
            height: search_box.height + dp(32)
            
            BoxLayout:
                id: search_box
                orientation: 'vertical'
                spacing: dp(12)
                size_hint_y: None
                height: self.minimum_height

                MDLabel:
                    text: '🔍 Search (100% Local & Private)'
                    font_style: 'H6'
                    bold: True
                    size_hint_y: None
                    height: self.texture_size[1]

                MDTextField:
                    id: query_input
                    hint_text: 'Enter search query...'
                    helper_text: '🔒 All searches are 100% local - files never leave your device'
                    helper_text_mode: 'persistent'
                    mode: 'rectangle'
                    size_hint_y: None
                    height: dp(70)
                    on_text: app.on_query_changed(self.text)

                BoxLayout:
                    size_hint_y: None
                    height: dp(48)
                    spacing: dp(8)

                    MDTextField:
                        id: limit_input
                        hint_text: 'Results'
                        text: '10'
                        input_filter: 'int'
                        mode: 'rectangle'
                        size_hint_x: 0.3

                    MDRaisedButton:
                        text: 'Search'
                        icon: 'magnify'
                        disabled: (not app.folder_selected) or (not app.query_ready) or app.loading
                        on_release: app.on_search_clicked()
                        md_bg_color: app.theme_cls.accent_color if not self.disabled else app.theme_cls.disabled_hint_text_color

        # Results Card
        ExpandableCard:
            height: dp(600) if app.has_results else dp(100)
            
            BoxLayout:
                orientation: 'vertical'
                spacing: dp(12)
                size_hint_y: None
                height: self.minimum_height

                MDLabel:
                    text: '📋 Results'
                    font_style: 'H6'
                    bold: True
                    size_hint_y: None
                    height: self.texture_size[1]

                MDLabel:
                    text: app.results_summary
                    theme_text_color: 'Secondary'
                    font_style: 'Caption'
                    size_hint_y: None
                    height: self.texture_size[1]
                    opacity: 1 if app.has_results else 0.5

                ScrollView:
                    size_hint_y: None
                    height: dp(500) if app.has_results else 0
                    do_scroll_x: False
                    bar_width: dp(8)
                    
                    GridLayout:
                        id: results_container
                        cols: 1
                        spacing: dp(12)
                        size_hint_y: None
                        height: self.minimum_height
                        padding: dp(4)
'''


from kivymd.uix.spinner import MDSpinner  # noqa: E402
from kivymd.uix.toolbar import MDTopAppBar  # noqa: E402
from kivymd.uix.card import MDCard  # noqa: E402
from kivymd.uix.textfield import MDTextField  # noqa: E402
from kivymd.uix.selectioncontrol import MDCheckbox  # noqa: E402
from kivy.uix.gridlayout import GridLayout  # noqa: E402
from kivy.uix.scrollview import ScrollView  # noqa: E402


class ResultRow(MDCard):
    file_name = StringProperty("")
    file_path = StringProperty("")
    score_text = StringProperty("")
    full_path = StringProperty("")


def _app_data_dir(app: MDApp) -> Path:
    return Path(app.user_data_dir)


def _copy_bundled_cache_if_present(dst_dir: Path) -> None:
    """Copy pre-fetched model cache from PyInstaller bundle to user_data_dir."""
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return

    src = Path(base) / "fastembed_cache"
    if not src.exists() or not src.is_dir():
        return

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
    status_text = StringProperty("Ready to index your files")
    selected_file_type = StringProperty('all')
    file_type_extensions = StringProperty('All supported: .txt .md .pdf .docx .png .jpg .jpeg .bmp .webp .tiff .tif')
    search_recursive = BooleanProperty(True)
    has_results = BooleanProperty(False)
    results_summary = StringProperty("No results yet")

    def build(self):
        self.title = "DocuFindLocal - Privacy-First Search"
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Teal"
        self.theme_cls.theme_style = "Light"
        self.root = Builder.load_string(KV)

        self._dialog: Optional[MDDialog] = None
        self._file_manager: Optional[MDFileManager] = None

        data_dir = _app_data_dir(self)
        self.db_path = data_dir / "local_index.sqlite3"
        self.cache_dir = data_dir / "model_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        _copy_bundled_cache_if_present(self.cache_dir)
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(self.cache_dir))

        self.indexer = LocalIndexer(db_path=self.db_path, cache_dir=self.cache_dir)
        self.searcher = LocalSearcher(db_path=self.db_path, cache_dir=self.cache_dir)
        return self.root

    def _set_status(self, text: str) -> None:
        self.status_text = text or ""

    def _show_error(self, text: str) -> None:
        if self._dialog:
            try:
                self._dialog.dismiss()
            except Exception:
                pass
        self._dialog = MDDialog(
            title="⚠️ DocuFindLocal",
            text=text,
            buttons=[MDRaisedButton(text="OK", on_release=lambda *_: self._dialog.dismiss())],
        )
        self._dialog.open()

    def _toast(self, text: str) -> None:
        self._set_status(text)
        try:
            from kivymd.uix.snackbar import Snackbar
            Snackbar(text=text).open()
        except Exception:
            pass

    def on_query_changed(self, text: str) -> None:
        self.query_ready = bool((text or "").strip())

    def set_file_type(self, ftype: str) -> None:
        self.selected_file_type = ftype
        if ftype == 'text':
            self.file_type_extensions = 'Text files: .txt .md'
        elif ftype == 'doc':
            self.file_type_extensions = 'Documents: .pdf .docx'
        elif ftype == 'image':
            self.file_type_extensions = 'Images: .png .jpg .jpeg .bmp .webp .tiff .tif'
        else:
            self.file_type_extensions = 'All supported: .txt .md .pdf .docx .png .jpg .jpeg .bmp .webp .tiff .tif'

    def set_search_recursive(self, active: bool) -> None:
        self.search_recursive = active

    def get_file_types(self) -> Optional[set[str]]:
        from docufind_local.local_search.text_extractors import SUPPORTED_TEXT_EXTS, SUPPORTED_DOC_EXTS, SUPPORTED_IMAGE_EXTS
        if self.selected_file_type == 'text':
            return SUPPORTED_TEXT_EXTS
        elif self.selected_file_type == 'doc':
            return SUPPORTED_DOC_EXTS
        elif self.selected_file_type == 'image':
            return SUPPORTED_IMAGE_EXTS
        else:
            return None

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
            title="🗑️ Clear local index?",
            text=(
                "This will delete the local search index database.\n\n"
                "You'll need to click 'Sync' again to rebuild it."
            ),
            buttons=[
                MDFlatButton(text="Cancel", on_release=_cancel),
                MDRaisedButton(text="Clear Index", on_release=_confirm),
            ],
        )
        self._dialog.open()

    def _start_clear_index(self) -> None:
        self.loading = True
        self.has_results = False
        self.results_summary = "No results yet"
        self.root.ids.results_container.clear_widgets()
        self._set_status("Clearing index...")
        threading.Thread(target=self._background_clear_index, daemon=True).start()

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

            self.indexer = LocalIndexer(db_path=self.db_path, cache_dir=self.cache_dir)
            self.searcher = LocalSearcher(db_path=self.db_path, cache_dir=self.cache_dir)

            msg = "✅ Index cleared. Please sync folder again." if deleted_any else "Index was already cleared."
            Clock.schedule_once(lambda dt, _m=msg: self._on_clear_index_done(_m), 0)
        except Exception as e:
            err = str(e)
            Clock.schedule_once(lambda dt, _e=err: self._on_clear_index_error(_e), 0)

    def _on_clear_index_done(self, msg: str) -> None:
        self.loading = False
        self._toast(msg)

    def _on_clear_index_error(self, err: str) -> None:
        self.loading = False
        self._show_error(f"Clear index failed: {err}")

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
        self._set_status("✅ Folder selected. Click 'Sync' to index files.")

    def on_sync_clicked(self) -> None:
        if not self.folder_selected:
            self._show_error("Please select a folder first.")
            return

        if not self.indexer.embedder.available and not self.indexer.clip.available:
            self._show_error(
                "No embedding backend is available.\n\n"
                f"Text embedder error: {self.indexer.embedder.init_error}\n"
                f"Image (CLIP) embedder error: {self.indexer.clip.init_error}\n\n"
                "Please install fastembed + onnxruntime."
            )
            return

        self.loading = True
        self.has_results = False
        self.results_summary = "No results yet"
        self.root.ids.results_container.clear_widgets()
        self._set_status("Syncing folder...")
        threading.Thread(target=self._background_sync, args=(self.selected_folder, self.search_recursive, self.get_file_types()), daemon=True).start()

    def _background_sync(self, folder: str, recursive: bool, file_types: Optional[set[str]]) -> None:
        try:
            def progress(msg: str) -> None:
                Clock.schedule_once(lambda dt, _m=msg: self._set_status(_m), 0)

            stats = self.indexer.index_folder(folder, progress=progress, recursive=recursive, file_types=file_types)
            final = f"✅ Sync complete! Indexed: {stats.indexed}, Skipped: {stats.skipped}, Failed: {stats.failed}"
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

    def on_search_clicked(self) -> None:
        if not self.folder_selected:
            self._show_error("Please select a folder first.")
            return
        query = (self.root.ids.query_input.text or "").strip()
        if not query:
            self._show_error("Please enter a search query.")
            return
        if not self.searcher.embedder.available and not self.searcher.clip.available:
            self._show_error(
                "No embedding backend is available.\n\n"
                f"Text embedder error: {self.searcher.embedder.init_error}\n"
                f"Image (CLIP) embedder error: {self.searcher.clip.init_error}\n\n"
                "Please install fastembed + onnxruntime."
            )
            return

        try:
            limit = int(self.root.ids.limit_input.text or 10)
        except Exception:
            limit = 10
        limit = max(1, min(50, limit))

        self.loading = True
        self.has_results = False
        self.results_summary = "Searching..."
        self.root.ids.results_container.clear_widgets()
        self._set_status("🔍 Searching...")
        threading.Thread(target=self._background_search, args=(self.selected_folder, query, limit, self.get_file_types()), daemon=True).start()

    def _background_search(self, folder: str, query: str, limit: int, file_types: Optional[set[str]]) -> None:
        try:
            results = self.searcher.search(folder=folder, query=query, limit=limit, file_types=file_types)
            Clock.schedule_once(lambda dt, _r=results: self._on_search_done(_r), 0)
        except Exception as e:
            err = str(e)
            Clock.schedule_once(lambda dt, _e=err: self._on_search_error(_e), 0)

    def _on_search_done(self, results) -> None:
        self.loading = False
        container = self.root.ids.results_container
        container.clear_widgets()
        
        if not results:
            self.has_results = False
            self.results_summary = "No results found. Have you synced the folder?"
            self._set_status("No results found")
            return

        self.has_results = True
        self.results_summary = f"Found {len(results)} result{'s' if len(results) != 1 else ''}"
        
        for r in results:
            name = Path(r.rel_path).name
            row = ResultRow()
            row.file_name = name
            row.file_path = r.rel_path
            row.score_text = f"Relevance score: {r.score:.3f}"
            row.full_path = r.path
            container.add_widget(row)
        
        self._set_status(f"✅ Found {len(results)} result{'s' if len(results) != 1 else ''}")

    def _on_search_error(self, err: str) -> None:
        self.loading = False
        self.has_results = False
        self.results_summary = "Search failed"
        self._show_error(f"Search failed: {err}")

    def open_result(self, full_path: str) -> None:
        if not full_path:
            return
        if platform == "android":
            self._set_status("Opening files is desktop-only for now")
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

    def open_file_location(self, full_path: str) -> None:
        """Open file location in explorer/finder with the file highlighted."""
        if not full_path:
            return
        if platform == "android":
            self._set_status("Opening file location is desktop-only for now")
            return
        try:
            p = Path(full_path)
            if not p.exists():
                self._show_error("File not found on disk")
                return
            
            if platform == "win":
                # Windows: Use explorer with /select flag to highlight the file
                import subprocess
                subprocess.run(['explorer', '/select,', str(p)])
            elif platform == "macosx":
                # macOS: Use 'open -R' to reveal in Finder
                os.system(f'open -R "{p}"')
            else:
                # Linux: Try various file managers
                # Most don't support highlighting, so open parent folder
                parent = p.parent
                try:
                    # Try nautilus (GNOME) with --select
                    import subprocess
                    result = subprocess.run(['nautilus', '--select', str(p)], 
                                          capture_output=True, timeout=2)
                    if result.returncode != 0:
                        raise Exception("Nautilus not available")
                except:
                    try:
                        # Try dolphin (KDE) with --select
                        subprocess.run(['dolphin', '--select', str(p)], timeout=2)
                    except:
                        # Fallback: just open the parent folder
                        os.system(f'xdg-open "{parent}"')
        except Exception as e:
            self._show_error(f"Could not open file location: {e}")


if __name__ == "__main__":
    DocuFindLocalApp().run()