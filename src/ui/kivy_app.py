"""
KivyMD UI for DocuFindAI

Cross-platform app (Desktop + Android) that calls the backend VectorStore.

Usage: run this module with the virtualenv active. On desktop the app will
try to open a native folder picker; on Android paste a path or use SAF if
available via plyer/filechooser.

This file integrates with the existing backend: it will attempt to call
`VectorStore.hybrid_search(query, n_results, path=...)` when available;
otherwise it will POST to the local API at `http://localhost:8000/search/`.

Requires: Kivy, KivyMD, requests, plyer (optional), and the project packages
to be importable (this script adds the project root to sys.path).

"""
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import StringProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage
from kivy.utils import platform

from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton, MDIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.list import MDList, OneLineAvatarIconListItem, ImageLeftWidget, OneLineListItem
from kivymd.uix.dialog import MDDialog
from kivymd.uix.textfield import MDTextField
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.menu import MDDropdownMenu

import threading
import os
import sys
import inspect
import json

# Ensure project `src` is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.database.vector_store import VectorStore

try:
    import requests
except Exception:
    requests = None

try:
    from plyer import filechooser
except Exception:
    filechooser = None


KV = '''
BoxLayout:
    orientation: 'vertical'
    padding: dp(12)
    spacing: dp(12)

    MDLabel:
        text: 'DocuFindAI'
        halign: 'center'
        font_style: 'H4'

    # Search query
    MDTextField:
        id: query_input
        hint_text: 'Search query'
        helper_text_mode: 'on_focus'
        size_hint_y: None
        height: dp(48)

    # Folder path + Browse
    BoxLayout:
        size_hint_y: None
        height: dp(48)
        spacing: dp(8)

        MDTextField:
            id: paste_path
            hint_text: 'Paste folder path (or use Browse)'
            size_hint_x: 0.9

        MDIconButton:
            id: browse_button
            icon: 'folder-search'
            on_release: app.open_folder_picker()

    # Filters
    BoxLayout:
        size_hint_y: None
        height: dp(48)
        spacing: dp(8)

        MDTextField:
            id: limit_input
            hint_text: 'Limit (results)'
            text: '10'
            size_hint_x: 0.2
            input_filter: 'int'

        MDTextField:
            id: content_type_input
            hint_text: 'Content type (all, document, image)'
            text: 'all'
            size_hint_x: 0.3
            readonly: True
            on_focus: if self.focus: app.open_content_type_menu(self)

        MDRaisedButton:
            id: search_btn
            text: 'Search'
            size_hint_x: 0.2
            on_release: app.on_search_clicked()

        MDRaisedButton:
            id: recent_btn
            text: 'Recent Paths'
            size_hint_x: 0.2
            on_release: app.show_recent_paths()

        MDSpinner:
            id: spinner
            size_hint: None, None
            size: dp(36), dp(36)
            active: False

    MDLabel:
        id: status_label
        text: ''
        halign: 'left'
        theme_text_color: 'Hint'

    ScrollView:
        MDList:
            id: results_list

'''


class ResultItem(OneLineAvatarIconListItem):
    """List item that optionally shows an image on the left."""

    def __init__(self, filename: str, snippet: str, score: float, thumbnail: str = None, **kwargs):
        super().__init__(text=f"{filename} — {score:.3f}", **kwargs)
        # Clicking copies snippet to clipboard
        self.filename = filename
        self.snippet = snippet
        if thumbnail:
            img = ImageLeftWidget()
            img.add_widget(AsyncImage(source=thumbnail))
            self.add_widget(img)

    def on_release(self):
        # copy snippet to clipboard
        if self.snippet:
            Clipboard.copy(self.snippet)


class DocuFindApp(MDApp):
    file_type_options = ['all', 'document', 'image']

    def build(self):
        self.title = 'DocuFindAI'
        self.theme_cls.primary_palette = 'Blue'
        self.root = Builder.load_string(KV)

        # Create VectorStore client instance (local direct integration)
        try:
            self.vs = VectorStore()
        except Exception as e:
            print('Warning: VectorStore init failed, will use API fallback:', e)
            self.vs = None

        # menu for content_type: use OneLineListItem viewclass
        menu_items = []
        for t in self.file_type_options:
            menu_items.append({
                "text": t,
                "viewclass": "OneLineListItem",
                "height": dp(48),
                "on_release": lambda x=t: self.set_content_type(x),
            })
        # set caller so menu can be positioned relative to the input
        self.content_menu = MDDropdownMenu(caller=self.root.ids.content_type_input, items=menu_items, width_mult=3)

        # Load recent paths
        self.recent_paths = self._load_recent_paths()

        return self.root

    def set_content_type(self, v):
        self.root.ids.content_type_input.text = v
        self.content_menu.dismiss()

    def open_content_type_menu(self, caller):
        self.content_menu.caller = caller
        self.content_menu.open()

    def open_folder_picker(self):
        """Open folder picker. On platforms without folder picker, user can paste path."""
        # Prefer folder selection. On Android filechooser may return file paths; use dirname.
        if filechooser:
            try:
                filechooser.open_file(on_selection=self._on_folder_selected)
                return
            except Exception:
                pass

        # Desktop: tkinter askdirectory
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            folder = filedialog.askdirectory()
            if folder:
                self.root.ids.paste_path.text = folder
                return
        except Exception:
            pass

        self.root.ids.status_label.text = 'Folder picker not available; paste path manually.'

    def _on_folder_selected(self, selection):
        if not selection:
            return
        path = selection[0] if isinstance(selection, (list, tuple)) else selection
        # if a file path was returned, convert to directory
        if os.path.isfile(path):
            path = os.path.dirname(path)
        self.root.ids.paste_path.text = path

    def show_recent_paths(self):
        # show modal dialog with recent paths
        if not self.recent_paths:
            self.root.ids.status_label.text = 'No recent paths saved.'
            return

        content = MDList()
        for p in self.recent_paths:
            # validate existence
            valid = os.path.exists(p)
            item = OneLineListItem(text=p, on_release=lambda x, path=p: self._select_recent_path(path))
            content.add_widget(item)

        self._recent_dialog = MDDialog(title='Recent Paths', type='custom', content_cls=content, size_hint=(0.9, 0.7))
        self._recent_dialog.open()

    def _select_recent_path(self, path):
        if os.path.exists(path):
            self.root.ids.paste_path.text = path
            try:
                self._recent_dialog.dismiss()
            except Exception:
                pass
        else:
            self.root.ids.status_label.text = 'Selected path does not exist.'

    def _save_recent_path(self, path: str):
        if not path:
            return
        if path in self.recent_paths:
            self.recent_paths.remove(path)
        self.recent_paths.insert(0, path)
        self.recent_paths = self.recent_paths[:10]
        try:
            with open(os.path.join(ROOT, '.recent_paths.json'), 'w', encoding='utf-8') as f:
                json.dump(self.recent_paths, f)
        except Exception:
            pass

    def _load_recent_paths(self):
        try:
            p = os.path.join(ROOT, '.recent_paths.json')
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def on_search_clicked(self):
        # Read inputs and validate
        query = (self.root.ids.query_input.text or '').strip()
        path = (self.root.ids.paste_path.text or '').strip()
        content_type = (self.root.ids.content_type_input.text or 'all').strip()
        try:
            limit = int(self.root.ids.limit_input.text or 10)
        except Exception:
            limit = 10

        if not query:
            self.root.ids.status_label.text = 'Please enter a search query.'
            return

        # Path optional but if provided ensure exists
        if path and not os.path.exists(path):
            self.root.ids.status_label.text = 'Invalid folder path.'
            return

        # show spinner, disable search button and clear results
        self.root.ids.spinner.active = True
        self.root.ids.status_label.text = 'Searching...'
        self.root.ids.results_list.clear_widgets()
        try:
            self.root.ids.browse_button.disabled = True
            self.root.ids.recent_btn.disabled = True
            self.root.ids.search_btn.disabled = True
        except Exception:
            pass

        # disable Search by setting the widget disabled via id (search button not named; reuse browse_button state to indicate running)
        thread = threading.Thread(target=self._background_search, args=(query, path, content_type, limit), daemon=True)
        thread.start()

    def _background_search(self, query, path, content_type, limit):
        """Background thread to perform search while keeping UI responsive."""
        try:
            results = None


            # If a path is provided -> force on-the-fly folder search (bypass ChromaDB)
            if path:
                # Prefer direct VectorStore call if it supports path parameter
                if self.vs is not None:
                    try:
                        sig = inspect.signature(self.vs.hybrid_search)
                        if 'path' in sig.parameters:
                            # try to pass content_type if supported
                            try:
                                results = self.vs.hybrid_search(query, n_results=limit, path=path, content_type=content_type)
                            except TypeError:
                                results = self.vs.hybrid_search(query, n_results=limit, path=path)
                        else:
                            results = None
                    except Exception:
                        results = None

                # Fallback to API call that supports on-the-fly folder search
                if results is None and requests:
                    try:
                        payload = {
                            'query': query,
                            'content_type': content_type,
                            'limit': limit,
                            'path': path
                        }
                        resp = requests.post('http://localhost:8000/search/', json=payload, timeout=120)
                        if resp.status_code == 200:
                            results = resp.json()
                        else:
                            results = {'error': f'Server error: {resp.status_code}'}
                    except Exception as e:
                        results = {'error': str(e)}

            else:
                # No path: use indexed search (ChromaDB)
                if self.vs is not None:
                    try:
                        results = self.vs.hybrid_search(query, n_results=limit)
                    except Exception as e:
                        results = {'error': str(e)}
                elif requests:
                    try:
                        resp = requests.post('http://localhost:8000/search/', json={
                            'query': query,
                            'content_type': content_type,
                            'limit': limit
                        }, timeout=30)
                        results = resp.json()
                    except Exception as e:
                        results = {'error': str(e)}

            # If still none, try persistent hybrid search via local API or direct call
            if results is None:
                if self.vs is not None:
                    try:
                        results = self.vs.hybrid_search(query, n_results=limit)
                    except Exception as e:
                        results = {'error': str(e)}
                elif requests:
                    try:
                        resp = requests.post('http://localhost:8000/search/', json={
                            'query': query,
                            'content_type': content_type,
                            'limit': limit
                        }, timeout=30)
                        results = resp.json()
                    except Exception as e:
                        results = {'error': str(e)}

            # Save recent path
            if path:
                self._save_recent_path(path)

            # schedule UI update
            Clock.schedule_once(lambda dt: self._display_results(results), 0)

        except Exception as e:
            Clock.schedule_once(lambda dt: self._display_results({'error': str(e)}), 0)

    def _display_results(self, results):
        # re-enable controls
        try:
            self.root.ids.browse_button.disabled = False
            self.root.ids.recent_btn.disabled = False
            self.root.ids.search_btn.disabled = False
        except Exception:
            pass

        self.root.ids.spinner.active = False
        rlabel = self.root.ids.status_label

        if results is None:
            rlabel.text = 'No results returned.'
            return

        if isinstance(results, dict) and results.get('error'):
            rlabel.text = f"Error: {results.get('error')}"
            return

        # Normalize possible response shapes
        items = []
        if isinstance(results, dict) and 'results' in results:
            items = results['results']
        elif isinstance(results, dict) and 'text_results' in results and 'image_results' in results:
            # combine hybrid_search dict
            items = []
            for t in results.get('text_results', []):
                items.append({'file': t.get('id') or t.get('document') or t.get('filename','text'), 'text': t.get('document') or t.get('metadata',{}).get('filename',''), 'score': t.get('score', 0)})
            for i in results.get('image_results', []):
                items.append({'file': i.get('id'), 'text': i.get('description',''), 'score': i.get('score', 0)})
        elif isinstance(results, list):
            items = results
        else:
            # unexpected shape
            rlabel.text = 'Unexpected response shape from backend.'
            return

        if not items:
            rlabel.text = 'No results found.'
            return

        rlabel.text = f'Returned {len(items)} results.'
        list_widget = self.root.ids.results_list

        for it in items:
            fname = it.get('file') or it.get('id') or 'unknown'
            text = it.get('text') or it.get('document') or ''
            score = it.get('score') or it.get('distance') or 0.0
            thumb = None
            # if metadata contains thumbnail or filepath, optionally set thumb
            md = it.get('metadata') or {}
            fp = md.get('filepath') or md.get('filename')
            if fp and os.path.exists(fp) and fp.lower().endswith(('.png', '.jpg', '.jpeg')):
                thumb = fp

            item = ResultItem(filename=fname, snippet=text, score=float(score), thumbnail=thumb)
            list_widget.add_widget(item)


if __name__ == '__main__':
    DocuFindApp().run()
