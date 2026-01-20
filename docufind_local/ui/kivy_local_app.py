"""DocuFindLocal Desktop (Local Search) - Responsive UI

100% local, privacy-safe desktop app with responsive design.
Includes first-run setup flow for downloading AI models.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty, StringProperty
from kivy.utils import platform
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition

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


# ---------------------------------------------------------------------------
# Model names for first-run download
# ---------------------------------------------------------------------------
TEXT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CLIP_VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"
CLIP_TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"


KV = r'''
#:import Window kivy.core.window.Window
#:import FadeTransition kivy.uix.screenmanager.FadeTransition

# ---------------------------------------------------------------------------
# First-Run Setup Screen
# ---------------------------------------------------------------------------
<FirstRunScreen>:
    name: 'first_run'
    
    MDCard:
        orientation: 'vertical'
        size_hint: 0.9, None
        height: self.minimum_height
        pos_hint: {'center_x': 0.5, 'center_y': 0.5}
        padding: dp(32)
        spacing: dp(20)
        radius: [dp(16)]
        elevation: 4
        md_bg_color: app.theme_cls.bg_normal
        
        MDLabel:
            text: '🔍 Welcome to DocuFindLocal'
            font_style: 'H5'
            bold: True
            halign: 'center'
            size_hint_y: None
            height: self.texture_size[1]
        
        MDLabel:
            text: 'DocuFindLocal runs 100% locally to protect your privacy.'
            halign: 'center'
            theme_text_color: 'Secondary'
            size_hint_y: None
            height: self.texture_size[1]
        
        MDCard:
            orientation: 'vertical'
            padding: dp(16)
            spacing: dp(8)
            size_hint_y: None
            height: self.minimum_height
            radius: [dp(8)]
            md_bg_color: app.theme_cls.bg_light
            
            MDLabel:
                text: 'On first launch, it needs to download AI models required for:'
                size_hint_y: None
                height: self.texture_size[1]
                text_size: self.width, None
            
            MDLabel:
                text: '• Text search'
                theme_text_color: 'Secondary'
                size_hint_y: None
                height: self.texture_size[1]
            
            MDLabel:
                text: '• Image search'
                theme_text_color: 'Secondary'
                size_hint_y: None
                height: self.texture_size[1]
        
        MDCard:
            orientation: 'vertical'
            padding: dp(12)
            spacing: dp(4)
            size_hint_y: None
            height: self.minimum_height
            radius: [dp(8)]
            md_bg_color: [0.9, 0.95, 0.9, 1]
            
            MDLabel:
                text: '✔ One-time setup'
                theme_text_color: 'Custom'
                text_color: [0.2, 0.5, 0.2, 1]
                size_hint_y: None
                height: self.texture_size[1]
            
            MDLabel:
                text: '✔ No files are uploaded'
                theme_text_color: 'Custom'
                text_color: [0.2, 0.5, 0.2, 1]
                size_hint_y: None
                height: self.texture_size[1]
            
            MDLabel:
                text: '✔ Works fully offline after setup'
                theme_text_color: 'Custom'
                text_color: [0.2, 0.5, 0.2, 1]
                size_hint_y: None
                height: self.texture_size[1]
        
        # Progress section (only visible during download)
        BoxLayout:
            orientation: 'vertical'
            size_hint_y: None
            height: dp(60) if app.setup_in_progress else 0
            opacity: 1 if app.setup_in_progress else 0
            spacing: dp(8)
            
            BoxLayout:
                size_hint_y: None
                height: dp(32)
                spacing: dp(12)
                
                MDSpinner:
                    size_hint: None, None
                    size: dp(28), dp(28)
                    active: app.setup_in_progress
                
                MDLabel:
                    text: app.setup_status
                    theme_text_color: 'Primary'
                    valign: 'center'
        
        # Buttons
        BoxLayout:
            orientation: 'vertical' if Window.width < dp(500) else 'horizontal'
            size_hint_y: None
            height: dp(100) if Window.width < dp(500) else dp(50)
            spacing: dp(12)
            
            MDRaisedButton:
                text: 'Download Models (Recommended)'
                size_hint_x: 1
                disabled: app.setup_in_progress
                md_bg_color: app.theme_cls.primary_color if not self.disabled else [0.7, 0.7, 0.7, 1]
                on_release: app.start_model_download()
            
            MDFlatButton:
                text: 'Exit'
                size_hint_x: 1 if Window.width < dp(500) else 0.4
                disabled: app.setup_in_progress
                on_release: app.stop()

# ---------------------------------------------------------------------------
# Main Application Screen
# ---------------------------------------------------------------------------
<MainScreen>:
    name: 'main'
    
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
                right_action_items: [['help-circle', lambda x: app.open_guide_dialog()]]

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
                            md_bg_color: [0.2, 0.6, 0.86, 1] if app.selected_file_type == 'all' else [0.6, 0.6, 0.6, 1]

                        MDRaisedButton:
                            text: 'Text'
                            on_release: app.set_file_type('text')
                            md_bg_color: [0.2, 0.6, 0.86, 1] if app.selected_file_type == 'text' else [0.6, 0.6, 0.6, 1]

                        MDRaisedButton:
                            text: 'Documents'
                            on_release: app.set_file_type('doc')
                            md_bg_color: [0.2, 0.6, 0.86, 1] if app.selected_file_type == 'doc' else [0.6, 0.6, 0.6, 1]

                        MDRaisedButton:
                            text: 'Images'
                            on_release: app.set_file_type('image')
                            md_bg_color: [0.2, 0.6, 0.86, 1] if app.selected_file_type == 'image' else [0.6, 0.6, 0.6, 1]

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
                height: results_box.height + dp(32)
                
                BoxLayout:
                    id: results_box
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

                    GridLayout:
                        id: results_container
                        cols: 1
                        spacing: dp(12)
                        size_hint_y: None
                        height: self.minimum_height
                        padding: dp(4)

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

# ---------------------------------------------------------------------------
# Root ScreenManager
# ---------------------------------------------------------------------------
ScreenManager:
    id: screen_manager
    transition: FadeTransition()
    
    FirstRunScreen:
        id: first_run_screen
    
    MainScreen:
        id: main_screen
'''


from kivymd.uix.spinner import MDSpinner  # noqa: E402
from kivymd.uix.toolbar import MDTopAppBar  # noqa: E402
from kivymd.uix.card import MDCard  # noqa: E402
from kivymd.uix.textfield import MDTextField  # noqa: E402
from kivymd.uix.selectioncontrol import MDCheckbox  # noqa: E402
from kivy.uix.gridlayout import GridLayout  # noqa: E402
from kivy.uix.scrollview import ScrollView  # noqa: E402


# ---------------------------------------------------------------------------
# Screen Classes
# ---------------------------------------------------------------------------
class FirstRunScreen(Screen):
    """First-run setup screen shown when models need to be downloaded."""
    pass


class MainScreen(Screen):
    """Main application screen with search functionality."""
    pass


class ResultRow(MDCard):
    file_name = StringProperty("")
    file_path = StringProperty("")
    score_text = StringProperty("")
    full_path = StringProperty("")


def _app_data_dir(app: MDApp) -> Path:
    return Path(app.user_data_dir)


def _get_bundled_cache_path() -> Optional[Path]:
    """Get the path to bundled fastembed_cache in PyInstaller builds.
    
    For onefile builds: sys._MEIPASS / fastembed_cache
    For onedir builds: executable directory / fastembed_cache
    """
    print("\n========== DIAGNOSTIC: _get_bundled_cache_path() ==========")
    print(f"sys.frozen: {getattr(sys, 'frozen', False)}")
    print(f"sys.executable: {sys.executable}")
    
    # Check for onefile build (_MEIPASS is the temp extraction dir)
    meipass = getattr(sys, "_MEIPASS", None)
    print(f"sys._MEIPASS: {meipass}")
    if meipass:
        candidate = Path(meipass) / "fastembed_cache"
        print(f"Checking onefile candidate: {candidate}")
        print(f"  Exists: {candidate.exists()}, Is dir: {candidate.is_dir() if candidate.exists() else 'N/A'}")
        if candidate.exists() and candidate.is_dir():
            print(f"  ✓ Found bundled cache (onefile): {candidate}")
            return candidate
    
    # Check for onedir build (files are next to the executable)
    if getattr(sys, "frozen", False):
        # sys.executable points to the .exe in frozen builds
        exe_dir = Path(sys.executable).parent
        candidate = exe_dir / "fastembed_cache"
        print(f"Checking onedir candidate: {candidate}")
        print(f"  Exists: {candidate.exists()}, Is dir: {candidate.is_dir() if candidate.exists() else 'N/A'}")
        if candidate.exists() and candidate.is_dir():
            print(f"  ✓ Found bundled cache (onedir): {candidate}")
            return candidate
    
    print("  ✗ No bundled cache found")
    print("=========================================================\n")
    return None


def _copy_bundled_cache_if_present(dst_dir: Path) -> None:
    """Copy pre-fetched model cache from PyInstaller bundle to user_data_dir.
    
    Works for both onefile and onedir PyInstaller builds.
    """
    print("\n========== DIAGNOSTIC: _copy_bundled_cache_if_present() ==========")
    print(f"Destination dir: {dst_dir}")
    
    src = _get_bundled_cache_path()
    if not src:
        print("No bundled cache source found - skipping copy")
        print("=================================================================\n")
        return

    # Skip if destination already has content (models already cached)
    print(f"Destination exists: {dst_dir.exists()}")
    if dst_dir.exists():
        contents = list(dst_dir.iterdir())
        print(f"Destination contents ({len(contents)} items): {[p.name for p in contents[:10]]}")
        if any(dst_dir.iterdir()):
            print("Destination already has content - skipping copy")
            print("=================================================================\n")
            return

    print(f"Copying bundled cache from {src} to {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst_dir / item.name
        print(f"  Copying: {item.name}")
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    print("✓ Bundled cache copied successfully")
    print("=================================================================\n")


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
    
    # First-run setup properties
    setup_in_progress = BooleanProperty(False)
    setup_status = StringProperty("")

    def build(self):
        print("\n" + "="*80)
        print("DIAGNOSTIC: DocuFindLocalApp.build() starting")
        print("="*80)
        
        self.title = "DocuFindLocal - Privacy-First Search"
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Teal"
        self.theme_cls.theme_style = "Light"
        self.root = Builder.load_string(KV)

        self._dialog: Optional[MDDialog] = None
        self._file_manager: Optional[MDFileManager] = None

        # Initialize paths
        data_dir = _app_data_dir(self)
        print(f"\nApp data directory: {data_dir}")
        self.db_path = data_dir / "local_index.sqlite3"
        self.cache_dir = data_dir / "model_cache"
        print(f"Database path: {self.db_path}")
        print(f"Cache directory: {self.cache_dir}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Copy bundled cache if present (for PyInstaller builds)
        _copy_bundled_cache_if_present(self.cache_dir)
        
        # Check environment variables BEFORE setting
        print(f"\nEnvironment variables BEFORE setting:")
        print(f"  HF_HOME: {os.environ.get('HF_HOME', 'NOT SET')}")
        print(f"  TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'NOT SET')}")
        print(f"  HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE', 'NOT SET')}")
        print(f"  FASTEMBED_CACHE_PATH: {os.environ.get('FASTEMBED_CACHE_PATH', 'NOT SET')}")
        
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(self.cache_dir))
        
        print(f"\nEnvironment variables AFTER setting:")
        print(f"  HF_HOME: {os.environ.get('HF_HOME', 'NOT SET')}")
        print(f"  TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'NOT SET')}")
        print(f"  HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE', 'NOT SET')}")
        print(f"  FASTEMBED_CACHE_PATH: {os.environ.get('FASTEMBED_CACHE_PATH', 'NOT SET')}")

        # Indexer/Searcher will be initialized after first-run check
        self.indexer = None
        self.searcher = None
        
        print("\nDocuFindLocalApp.build() completed")
        print("="*80 + "\n")

        return self.root

    def on_start(self):
        """Called after build() - check if first run is needed."""
        print("\n" + "="*80)
        print("DIAGNOSTIC: DocuFindLocalApp.on_start()")
        print("="*80)
        
        is_first = self._is_first_run()
        print(f"\n_is_first_run() returned: {is_first}")
        
        if is_first:
            print("→ Showing first-run screen")
            # Show first-run screen
            self.root.current = 'first_run'
        else:
            print("→ Going directly to main app (models exist)")
            # Models exist, go directly to main app
            self._initialize_backends()
            self.root.current = 'main'
        
        print("="*80 + "\n")

    def _is_first_run(self) -> bool:
        """Check if this is the first run by looking for model files in cache."""
        print("\n========== DIAGNOSTIC: _is_first_run() ==========")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Cache dir exists: {self.cache_dir.exists()}")
        
        if not self.cache_dir.exists():
            print("✗ Cache directory does not exist → FIRST RUN")
            print("================================================\n")
            return True
        
        # Check if cache directory has any model subdirectories
        # fastembed creates directories like "models--BAAI--bge-small-en-v1.5"
        try:
            contents = list(self.cache_dir.iterdir())
            print(f"\nCache directory contents ({len(contents)} items):")
            for i, p in enumerate(contents):
                if i < 20:  # Show first 20 items
                    print(f"  - {p.name} ({'DIR' if p.is_dir() else 'FILE'})")
                elif i == 20:
                    print(f"  ... and {len(contents) - 20} more items")
            
            # Look for model directories (not just any file)
            model_dirs = [p for p in contents if p.is_dir() and (
                p.name.startswith("models--") or 
                p.name.startswith("onnx") or
                "bge" in p.name.lower() or
                "clip" in p.name.lower()
            )]
            
            print(f"\nFound {len(model_dirs)} model directories:")
            for d in model_dirs:
                print(f"  ✓ {d.name}")
            
            result = len(model_dirs) == 0
            print(f"\nResult: {'FIRST RUN (no models)' if result else 'NOT FIRST RUN (models exist)'}")
            print("================================================\n")
            return result
        except Exception as e:
            print(f"✗ Exception while checking cache: {e}")
            print("================================================\n")
            return True

    def _initialize_backends(self) -> None:
        """Initialize indexer and searcher backends."""
        from docufind_local.local_search.indexer import LocalIndexer
        from docufind_local.local_search.searcher import LocalSearcher
        
        self.indexer = LocalIndexer(db_path=self.db_path, cache_dir=self.cache_dir)
        self.searcher = LocalSearcher(db_path=self.db_path, cache_dir=self.cache_dir)

    # ---------------------------------------------------------------------------
    # First-Run Model Download
    # ---------------------------------------------------------------------------
    def start_model_download(self) -> None:
        """Start downloading models in background thread."""
        self.setup_in_progress = True
        self.setup_status = "Preparing download..."
        threading.Thread(target=self._background_download_models, daemon=True).start()

    def _background_download_models(self) -> None:
        """Download fastembed models in background. Updates UI via Clock."""
        try:
            os.environ.setdefault("FASTEMBED_CACHE_PATH", str(self.cache_dir))
            
            # Step 1: Download text embedding model
            Clock.schedule_once(
                lambda dt: self._update_setup_status("Downloading text search model..."), 0
            )
            
            from fastembed import TextEmbedding
            
            # This will download the model if not cached
            text_model = TextEmbedding(model_name=TEXT_MODEL_NAME)
            # Warm up with a test embed to ensure model is fully loaded
            list(text_model.embed(["test"]))
            del text_model
            
            Clock.schedule_once(
                lambda dt: self._update_setup_status("✓ Text model ready. Downloading image search model..."), 0
            )
            
            # Step 2: Download CLIP vision model
            from fastembed import ImageEmbedding
            
            Clock.schedule_once(
                lambda dt: self._update_setup_status("Downloading image search model (vision)..."), 0
            )
            
            vision_model = ImageEmbedding(model_name=CLIP_VISION_MODEL)
            del vision_model
            
            # Step 3: Download CLIP text model
            Clock.schedule_once(
                lambda dt: self._update_setup_status("Downloading image search model (text)..."), 0
            )
            
            clip_text = TextEmbedding(model_name=CLIP_TEXT_MODEL)
            list(clip_text.embed(["test"]))
            del clip_text
            
            Clock.schedule_once(
                lambda dt: self._update_setup_status("✓ All models downloaded!"), 0
            )
            
            # Success - schedule transition to main app
            Clock.schedule_once(lambda dt: self._on_download_complete(), 0.5)
            
        except Exception as e:
            error_msg = str(e)
            Clock.schedule_once(lambda dt, err=error_msg: self._on_download_error(err), 0)

    def _update_setup_status(self, status: str) -> None:
        """Update setup status text (called from main thread)."""
        self.setup_status = status

    def _on_download_complete(self) -> None:
        """Called when model download completes successfully."""
        self.setup_in_progress = False
        self.setup_status = "Setup complete!"
        
        # Initialize backends with downloaded models
        self._initialize_backends()
        
        # Transition to main screen
        self.root.current = 'main'

    def _on_download_error(self, error: str) -> None:
        """Called when model download fails."""
        self.setup_in_progress = False
        self.setup_status = ""
        
        # Show error dialog with retry option
        if self._dialog:
            try:
                self._dialog.dismiss()
            except Exception:
                pass
        
        def _retry(*_args):
            if self._dialog:
                self._dialog.dismiss()
            self.start_model_download()
        
        def _exit(*_args):
            if self._dialog:
                self._dialog.dismiss()
            self.stop()
        
        self._dialog = MDDialog(
            title="⚠️ Download Failed",
            text=(
                f"Could not download AI models:\n\n{error}\n\n"
                "Please check your internet connection and try again."
            ),
            buttons=[
                MDFlatButton(text="Exit", on_release=_exit),
                MDRaisedButton(text="Retry", on_release=_retry),
            ],
        )
        self._dialog.open()

    # ---------------------------------------------------------------------------
    # Helper to access MainScreen IDs
    # ---------------------------------------------------------------------------
    @property
    def main_screen(self):
        """Get the main screen widget."""
        return self.root.ids.main_screen
    
    def _get_main_ids(self):
        """Get the IDs from the main screen's ScrollView content."""
        # The MainScreen contains a ScrollView > BoxLayout with the IDs
        main = self.root.ids.main_screen
        # Walk children to find the BoxLayout with our IDs
        for child in main.walk():
            if hasattr(child, 'ids') and 'results_container' in child.ids:
                return child.ids
        # Fallback: try direct ids on main screen children
        scrollview = main.children[0] if main.children else None
        if scrollview and scrollview.children:
            boxlayout = scrollview.children[0]
            return boxlayout.ids if hasattr(boxlayout, 'ids') else {}
        return {}

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

    def open_guide_dialog(self) -> None:
        """Show a short usage guide explaining how to run the app."""
        if self._dialog:
            try:
                self._dialog.dismiss()
            except Exception:
                pass

        guide_text = (
            "How to use DocuFindLocal:\n\n"
            "1) Click 'Browse' and choose a folder which contains documents to search.\n"
            "(This folder will be the destination scanned for documents.)\n\n"
            "2) Click 'Sync' to index files in the chosen folder.\n\n"
            "3) Choose the file type filter (Documents, Images, Text or All).\n"
            "If unsure, select 'All'.\n\n"
            "4) (Optional) Check 'Search subfolders recursively' to include files in subdirectories.\n\n"
            "5) Enter a search query relevant to the document you want to find.\n\n"
            "6) Optionally set the number of results to return. Best value: 3\n\n"
            "7) Click 'Search' and view matching results in the Results section.\n"
        )

        self._dialog = MDDialog(
            title="Guide — How to run",
            text=guide_text,
            buttons=[MDFlatButton(text="OK", on_release=lambda *_: self._dialog.dismiss())],
        )
        self._dialog.open()

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
        main_ids = self._get_main_ids()
        if 'results_container' in main_ids:
            main_ids.results_container.clear_widgets()
        self._set_status("Clearing index...")
        threading.Thread(target=self._background_clear_index, daemon=True).start()

    def _background_clear_index(self) -> None:
        from docufind_local.local_search.indexer import LocalIndexer
        from docufind_local.local_search.searcher import LocalSearcher
        
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
        main_ids = self._get_main_ids()
        if 'results_container' in main_ids:
            main_ids.results_container.clear_widgets()
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
        main_ids = self._get_main_ids()
        query = (main_ids.query_input.text if 'query_input' in main_ids else "").strip()
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
            limit = int(main_ids.limit_input.text if 'limit_input' in main_ids else 10)
        except Exception:
            limit = 10
        limit = max(1, min(50, limit))

        self.loading = True
        self.has_results = False
        self.results_summary = "Searching..."
        if 'results_container' in main_ids:
            main_ids.results_container.clear_widgets()
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
        main_ids = self._get_main_ids()
        container = main_ids.results_container if 'results_container' in main_ids else None
        if container:
            container.clear_widgets()
        
        if not results:
            self.has_results = False
            self.results_summary = "No results found. Have you synced the folder?"
            self._set_status("No results found")
            return

        self.has_results = True
        self.results_summary = f"Found {len(results)} result{'s' if len(results) != 1 else ''}"
        
        if container:
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