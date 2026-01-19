"""DocuFindLocal Desktop (Local Search)

100% local, privacy-safe desktop app:
- User selects a folder
- Sync indexes supported files into local SQLite (text extraction + OCR + embeddings)
- Search runs locally over cached embeddings

Backend is NOT required.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.metrics import dp
from kivy.properties import BooleanProperty, StringProperty, NumericProperty, ListProperty
from kivy.utils import platform, get_color_from_hex

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

# Lazy-loaded backend imports - loaded AFTER UI renders
LocalIndexer = None
LocalSearcher = None


def _kv_rgba(hexstr: str):
    """KV helper: safe conversion from hex string to RGBA list."""
    if hexstr is None:
        return [0.0, 0.0, 0.0, 1.0]
    try:
        color = get_color_from_hex(str(hexstr))
        if color is None:
            return [0.0, 0.0, 0.0, 1.0]
        # Ensure it's a list and has 4 elements
        color_list = list(color) if color else [0.0, 0.0, 0.0, 1.0]
        if len(color_list) < 4:
            color_list = color_list + [1.0] * (4 - len(color_list))
        return color_list
    except Exception:
        # Fallback to black if anything goes wrong
        return [0.0, 0.0, 0.0, 1.0]


KV = r'''
#:import rgba docufind_local.ui.kivy_local_app._kv_rgba
#:import colors kivymd.color_definitions.colors
#:import Window kivy.core.window.Window

<SettingsContent@MDBoxLayout>:
    orientation: 'vertical'
    spacing: dp(24)
    padding: dp(32)
    size_hint_y: None
    adaptive_height: True

    MDLabel:
        text: '⚙️ Indexing Settings'
        font_style: 'H4'
        size_hint_y: None
        height: dp(56)
        halign: 'left'
        bold: True
        theme_text_color: 'Custom'
        text_color: rgba('#6366F1')

    MDCard:
        orientation: 'vertical'
        padding: dp(24)
        spacing: dp(20)
        size_hint_y: None
        height: dp(160)
        md_bg_color: rgba('#F5F3FF')
        elevation: 0
        radius: [dp(16)]
        
        MDBoxLayout:
            orientation: 'horizontal'
            spacing: dp(20)
            size_hint_y: None
            height: dp(56)
            MDLabel:
                text: '🚀 Auto-index on startup'
                theme_text_color: 'Custom'
                text_color: rgba('#1F2937')
                font_style: 'Subtitle1'
                bold: True
                size_hint_x: 0.7
                halign: 'left'
                valign: 'center'
            MDSwitch:
                id: auto_index_switch
                active: app.settings_auto_index
                size_hint_x: 0.3
                thumb_color_active: rgba('#6366F1')
                track_color_active: rgba('#A5B4FC')

        MDBoxLayout:
            orientation: 'horizontal'
            spacing: dp(20)
            size_hint_y: None
            height: dp(56)
            MDLabel:
                text: '📁 Include subfolders'
                theme_text_color: 'Custom'
                text_color: rgba('#1F2937')
                font_style: 'Subtitle1'
                bold: True
                size_hint_x: 0.7
                halign: 'left'
                valign: 'center'
            MDSwitch:
                id: include_subfolders_switch
                active: app.settings_include_subfolders
                size_hint_x: 0.3
                thumb_color_active: rgba('#6366F1')
                track_color_active: rgba('#A5B4FC')

    Widget:
        size_hint_y: None
        height: dp(16)

    MDLabel:
        text: '📄 File types to index'
        font_style: 'H6'
        size_hint_y: None
        height: dp(36)
        halign: 'left'
        bold: True
        theme_text_color: 'Custom'
        text_color: rgba('#6366F1')

    MDTextField:
        id: filetypes_input
        hint_text: 'e.g., PDF, DOCX, TXT, PNG, JPG'
        text: app.settings_file_types
        mode: 'round'
        fill_color: rgba('#F9FAFB')
        size_hint_y: None
        height: dp(68)
        font_size: dp(16)
        helper_text: 'Comma-separated file extensions'
        helper_text_mode: 'on_focus'
        line_color_normal: rgba('#E5E7EB')
        line_color_focus: rgba('#6366F1')

    Widget:
        size_hint_y: None
        height: dp(16)

    MDLabel:
        text: '🎯 Maximum search results'
        font_style: 'H6'
        size_hint_y: None
        height: dp(36)
        halign: 'left'
        bold: True
        theme_text_color: 'Custom'
        text_color: rgba('#6366F1')

    MDTextField:
        id: max_results_input
        hint_text: '10'
        text: app.settings_max_results
        mode: 'round'
        fill_color: rgba('#F9FAFB')
        input_filter: 'int'
        size_hint_y: None
        height: dp(68)
        font_size: dp(16)
        helper_text: 'Number of results to display'
        helper_text_mode: 'on_focus'
        line_color_normal: rgba('#E5E7EB')
        line_color_focus: rgba('#6366F1')

    Widget:
        size_hint_y: None
        height: dp(24)

    MDCard:
        orientation: 'vertical'
        padding: dp(24)
        spacing: dp(16)
        size_hint_y: None
        height: dp(132)
        md_bg_color: rgba('#FEF2F2')
        elevation: 0
        radius: [dp(16)]
        
        MDLabel:
            text: '⚠️ Danger Zone'
            font_style: 'H6'
            theme_text_color: 'Custom'
            text_color: rgba('#DC2626')
            size_hint_y: None
            height: dp(36)
            halign: 'left'
            bold: True
        
        MDRaisedButton:
            id: reset_button
            text: '🗑️ Clear All Index Data'
            size_hint_y: None
            height: dp(56)
            md_bg_color: rgba('#DC2626')
            elevation: 0
            font_size: dp(15)
            bold: True
            radius: [dp(12)]
            on_release: app.on_clear_index_clicked()


<ResultCard@MDCard>:
    orientation: 'vertical'
    padding: dp(28)
    spacing: dp(20)
    size_hint_y: None
    adaptive_height: True
    md_bg_color: rgba('#FFFFFF')
    elevation: 0
    radius: [dp(20)]
    ripple_behavior: True
    
    canvas.before:
        Color:
            rgba: rgba('#E5E7EB')
        Line:
            rounded_rectangle: self.x, self.y, self.width, self.height, dp(20), dp(20)
            width: dp(1.5)
    
    MDBoxLayout:
        orientation: 'horizontal'
        spacing: dp(20)
        size_hint_y: None
        height: dp(64)
        
        MDCard:
            size_hint: None, None
            size: dp(64), dp(64)
            md_bg_color: rgba('#EEF2FF')
            elevation: 0
            radius: [dp(16)]
            padding: dp(14)
            
            MDIcon:
                icon: app.icon_for_ext(root.file_ext)
                theme_text_color: 'Custom'
                text_color: rgba('#6366F1')
                font_size: dp(36)
                pos_hint: {'center_x': 0.5, 'center_y': 0.5}
        
        MDBoxLayout:
            orientation: 'vertical'
            spacing: dp(4)
            size_hint_x: 0.55
            
            MDLabel:
                id: file_name_label
                text: root.file_name
                font_style: 'H6'
                bold: True
                theme_text_color: 'Custom'
                text_color: rgba('#111827')
                shorten: True
                shorten_from: 'right'
                size_hint_y: None
                height: dp(28)
            
            MDLabel:
                text: '📄 Document File'
                font_style: 'Caption'
                theme_text_color: 'Custom'
                text_color: rgba('#6B7280')
                size_hint_y: None
                height: dp(20)
        
        Widget:
            size_hint_x: 0.05
        
        MDCard:
            size_hint: None, None
            size: dp(84), dp(56)
            md_bg_color: rgba('#10B981')
            elevation: 0
            radius: [dp(16)]
            padding: [dp(12), dp(6)]
            
            MDLabel:
                text: f'{root.score_percent}%'
                font_style: 'H5'
                theme_text_color: 'Custom'
                text_color: rgba('#FFFFFF')
                bold: True
                halign: 'center'
                valign: 'center'
    
    Widget:
        size_hint_y: None
        height: dp(8)
    
    MDCard:
        size_hint_y: None
        height: dp(12)
        md_bg_color: rgba('#F3F4F6')
        elevation: 0
        radius: [dp(6)]
        
        MDCard:
            size_hint_x: root.score_percent / 100
            md_bg_color: rgba('#10B981')
            elevation: 0
            radius: [dp(6)]
    
    MDCard:
        orientation: 'vertical'
        padding: dp(20)
        spacing: dp(10)
        size_hint_y: None
        height: dp(96)
        md_bg_color: rgba('#F9FAFB')
        elevation: 0
        radius: [dp(14)]
        
        MDLabel:
            text: '👁️ Preview'
            font_style: 'Subtitle1'
            theme_text_color: 'Custom'
            text_color: rgba('#6366F1')
            size_hint_y: None
            height: dp(24)
            halign: 'left'
            bold: True
        
        MDLabel:
            id: snippet_label
            text: root.snippet
            theme_text_color: 'Custom'
            text_color: rgba('#4B5563')
            font_style: 'Body1'
            size_hint_y: None
            text_size: self.width, None
            shorten: True
            max_lines: 2
            height: self.texture_size[1]
            markup: True
            halign: 'left'
    
    Widget:
        size_hint_y: None
        height: dp(12)
    
    MDBoxLayout:
        orientation: 'horizontal'
        spacing: dp(16)
        size_hint_y: None
        height: dp(56)
        
        MDRaisedButton:
            id: open_folder_btn
            text: '📁 Show in Folder'
            size_hint_x: 0.48
            height: dp(56)
            md_bg_color: rgba('#F3F4F6')
            theme_text_color: 'Custom'
            text_color: rgba('#374151')
            elevation: 0
            font_size: dp(15)
            bold: True
            radius: [dp(12)]
            on_release: app.open_folder_location(root.full_path)
        
        Widget:
            size_hint_x: 0.04
        
        MDRaisedButton:
            id: open_file_btn
            text: '🚀 Open File'
            size_hint_x: 0.48
            height: dp(56)
            md_bg_color: rgba('#6366F1')
            theme_text_color: 'Custom'
            text_color: rgba('#FFFFFF')
            elevation: 0
            font_size: dp(15)
            bold: True
            radius: [dp(12)]
            on_release: app.open_result(root.full_path)


MDBoxLayout:
    orientation: 'vertical'
    md_bg_color: rgba('#F9FAFB')
    
    # Modern gradient header with glassmorphism
    MDCard:
        id: top_bar
        orientation: 'horizontal'
        size_hint_y: None
        height: dp(80)
        md_bg_color: rgba('#6366F1')
        elevation: 0
        padding: [dp(32), 0, dp(32), 0]
        
        MDBoxLayout:
            orientation: 'horizontal'
            spacing: dp(24)
            
            MDCard:
                size_hint: None, None
                size: dp(56), dp(56)
                md_bg_color: rgba('#FFFFFF')
                elevation: 0
                radius: [dp(14)]
                padding: dp(8)
                pos_hint: {'center_y': 0.5}
                
                FitImage:
                    source: 'docufind_local/ui/assets/logo.png'
                    pos_hint: {'center_x': 0.5, 'center_y': 0.5}
            
            MDBoxLayout:
                orientation: 'vertical'
                spacing: dp(4)
                size_hint_x: None
                width: dp(240)
                pos_hint: {'center_y': 0.5}
                
                MDLabel:
                    text: '🔍 DocuFind AI'
                    font_style: 'H4'
                    theme_text_color: 'Custom'
                    text_color: rgba('#FFFFFF')
                    bold: True
                    size_hint_y: None
                    height: dp(36)
                    halign: 'left'
                
                MDLabel:
                    text: 'Smart Document Discovery'
                    font_style: 'Subtitle2'
                    theme_text_color: 'Custom'
                    text_color: rgba('#E0E7FF')
                    size_hint_y: None
                    height: dp(22)
                    halign: 'left'
            
            Widget:
            
            MDCard:
                size_hint: None, None
                size: dp(180), dp(48)
                md_bg_color: rgba('#FFFFFF')
                elevation: 0
                radius: [dp(24)]
                padding: [dp(20), 0]
                pos_hint: {'center_y': 0.5}
                
                MDLabel:
                    id: status_indicator
                    text: app.status_indicator
                    font_style: 'Subtitle1'
                    theme_text_color: 'Custom'
                    text_color: rgba('#6366F1')
                    bold: True
                    halign: 'center'
                    valign: 'center'
            
            MDSpinner:
                size_hint: None, None
                size: dp(28), dp(28)
                active: app.loading
                opacity: 1 if app.loading else 0
                pos_hint: {'center_y': 0.5}
                color: rgba('#FFFFFF')

    BoxLayout:
        orientation: 'horizontal'
        spacing: 0
        
        # Modern sidebar
        MDCard:
            orientation: 'vertical'
            size_hint_x: None
            width: dp(280)
            md_bg_color: rgba('#FFFFFF')
            padding: dp(24)
            spacing: dp(16)
            elevation: 8
            
            MDCard:
                orientation: 'vertical'
                padding: dp(12)
                size_hint_y: None
                height: dp(60)
                md_bg_color: rgba('#EDE7F6')
                elevation: 0
                radius: [dp(12)]
                
                MDLabel:
                    text: '⚡ QUICK ACTIONS'
                    font_style: 'H6'
                    theme_text_color: 'Custom'
                    text_color: rgba('#5E35B1')
                    size_hint_y: None
                    height: dp(24)
                    halign: 'center'
                    bold: True
                
                MDLabel:
                    text: 'Get started quickly'
                    font_style: 'Caption'
                    theme_text_color: 'Secondary'
                    size_hint_y: None
                    height: dp(16)
                    halign: 'center'
            
            MDBoxLayout:
                orientation: 'vertical'
                spacing: dp(12)
                size_hint_y: None
                height: dp(300)
                
                MDRaisedButton:
                    id: select_folder_btn
                    text: '📁 Select Folder'
                    size_hint_y: None
                    height: dp(60)
                    md_bg_color: rgba('#667EEA')
                    theme_text_color: 'Custom'
                    text_color: rgba('#FFFFFF')
                    elevation: 6
                    font_size: dp(16)
                    bold: True
                    on_release: app.open_folder_picker()
                
                MDRaisedButton:
                    id: reindex_btn
                    text: '🔄 Re-index Folder'
                    size_hint_y: None
                    height: dp(56)
                    md_bg_color: rgba('#E8EAF6')
                    theme_text_color: 'Custom'
                    text_color: rgba('#5E35B1')
                    elevation: 2
                    font_size: dp(15)
                    bold: True
                    disabled: (not app.folder_selected) or app.loading
                    on_release: app.on_sync_clicked()
                
                MDRaisedButton:
                    id: settings_btn
                    text: '⚙️ Settings'
                    size_hint_y: None
                    height: dp(56)
                    md_bg_color: rgba('#F5F5F5')
                    theme_text_color: 'Custom'
                    text_color: rgba('#424242')
                    elevation: 0
                    font_size: dp(15)
                    bold: True
                    on_release: app.show_settings_dialog()
                
                MDRaisedButton:
                    id: about_btn
                    text: 'ℹ️ About'
                    size_hint_y: None
                    height: dp(56)
                    md_bg_color: rgba('#F5F5F5')
                    theme_text_color: 'Custom'
                    text_color: rgba('#424242')
                    elevation: 0
                    font_size: dp(15)
                    bold: True
                    on_release: app.show_about_dialog()
            
            Widget:
            
            MDCard:
                orientation: 'vertical'
                padding: dp(16)
                size_hint_y: None
                height: dp(140)
                md_bg_color: rgba('#E8F5E9')
                elevation: 4
                radius: [dp(16)]
                spacing: dp(10)
                
                MDLabel:
                    text: '📂 Current Folder'
                    font_style: 'H6'
                    theme_text_color: 'Custom'
                    text_color: rgba('#2E7D32')
                    size_hint_y: None
                    height: dp(28)
                    halign: 'center'
                    bold: True
                
                Widget:
                    size_hint_y: None
                    height: dp(4)
                
                MDLabel:
                    id: folder_name_label
                    text: app.selected_folder_name or 'No folder selected'
                    font_style: 'Body2'
                    theme_text_color: 'Primary'
                    size_hint_y: None
                    height: dp(24)
                    shorten: True
                    shorten_from: 'right'
                    halign: 'left'
                
                MDLabel:
                    id: folder_path_label
                    text: app.selected_folder or 'Select a folder to begin'
                    font_style: 'Caption'
                    theme_text_color: 'Secondary'
                    size_hint_y: None
                    text_size: self.width, None
                    height: self.texture_size[1]
                    shorten: True
                    max_lines: 2
                    halign: 'left'
        
        # Main Content Area
        BoxLayout:
            orientation: 'vertical'
            padding: [dp(24), dp(16), dp(24), dp(16)]
            spacing: dp(20)
            
            # Hero search card
            MDCard:
                orientation: 'vertical'
                padding: dp(32)
                spacing: dp(20)
                size_hint_y: None
                height: dp(220)
                md_bg_color: rgba('#FFFFFF')
                elevation: 12
                radius: [dp(20)]
                
                MDLabel:
                    text: '🔍 Search Your Documents'
                    font_style: 'H3'
                    theme_text_color: 'Custom'
                    text_color: rgba('#1A237E')
                    bold: True
                    size_hint_y: None
                    height: dp(48)
                    halign: 'left'
                
                MDTextField:
                    id: query_input
                    hint_text: 'Search by meaning... Try "project timeline" or "budget report 2024"'
                    mode: 'fill'
                    fill_color: rgba('#F8F9FF')
                    size_hint_y: None
                    height: dp(68)
                    font_size: dp(17)
                    on_text: app.on_query_changed(self.text)
                    icon_right: 'magnify'
                    icon_right_color: rgba('#667EEA')
                
                MDBoxLayout:
                    orientation: 'horizontal'
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(60)
                    
                    MDCard:
                        orientation: 'horizontal'
                        size_hint_x: 0.35
                        md_bg_color: rgba('#F8F9FF')
                        elevation: 0
                        radius: [dp(12)]
                        padding: [dp(16), 0]
                        
                        MDLabel:
                            text: '📊 Max results:'
                            font_style: 'Subtitle1'
                            theme_text_color: 'Primary'
                            bold: True
                            size_hint_x: 0.6
                            halign: 'left'
                            valign: 'center'
                        
                        MDTextField:
                            id: limit_input
                            hint_text: '10'
                            text: '10'
                            input_filter: 'int'
                            size_hint_x: 0.4
                            mode: 'fill'
                            fill_color: rgba('#FFFFFF')
                            height: dp(60)
                            font_size: dp(16)
                    
                    Widget:
                    
                    MDRaisedButton:
                        id: search_button
                        text: '🚀 START SEARCH'
                        size_hint_x: None
                        width: dp(200)
                        height: dp(60)
                        disabled: (not app.folder_selected) or (not app.query_ready) or app.loading
                        md_bg_color: rgba('#667EEA')
                        theme_text_color: 'Custom'
                        text_color: rgba('#FFFFFF')
                        elevation: 8
                        font_size: dp(17)
                        bold: True
                        on_release: app.on_search_clicked()
            
            # Modern progress card
            MDCard:
                id: progress_card
                orientation: 'vertical'
                padding: dp(28)
                spacing: dp(16)
                size_hint_y: None
                height: dp(150) if app.loading and app.indexing_progress else 0
                opacity: 1 if (app.loading and app.indexing_progress) else 0
                disabled: not (app.loading and app.indexing_progress)
                md_bg_color: rgba('#E8F5E9')
                elevation: 8
                radius: [dp(20)]
                
                MDBoxLayout:
                    orientation: 'horizontal'
                    spacing: dp(16)
                    size_hint_y: None
                    height: dp(40)
                    
                    MDCard:
                        size_hint: None, None
                        size: dp(40), dp(40)
                        md_bg_color: rgba('#00D9A5')
                        elevation: 4
                        radius: [dp(10)]
                        padding: dp(8)
                        
                        MDIcon:
                            icon: 'sync'
                            theme_text_color: 'Custom'
                            text_color: rgba('#FFFFFF')
                            font_size: dp(24)
                            pos_hint: {'center_x': 0.5, 'center_y': 0.5}
                    
                    MDLabel:
                        text: app.indexing_progress
                        font_style: 'H6'
                        theme_text_color: 'Custom'
                        text_color: rgba('#2E7D32')
                        bold: True
                        halign: 'left'
                
                MDCard:
                    size_hint_y: None
                    height: dp(14)
                    md_bg_color: rgba('#C8E6C9')
                    elevation: 0
                    radius: [dp(7)]
                    
                    MDCard:
                        size_hint_x: app.indexing_progress_value / 100
                        md_bg_color: rgba('#00D9A5')
                        elevation: 0
                        radius: [dp(7)]
                
                MDLabel:
                    text: app.status_text
                    font_style: 'Caption'
                    theme_text_color: 'Secondary'
                    halign: 'center'
                    size_hint_y: None
                    height: dp(20)
            
            # Results Section
            MDBoxLayout:
                orientation: 'vertical'
                spacing: dp(8)
                
                MDBoxLayout:
                    orientation: 'horizontal'
                    spacing: dp(8)
                    size_hint_y: None
                    height: dp(40)
                    
                    MDLabel:
                        text: 'Search Results'
                        font_style: 'H6'
                        theme_text_color: 'Primary'
                        size_hint_x: None
                        width: dp(140)
                        halign: 'left'
                    
                    MDLabel:
                        id: results_count_label
                        text: ''
                        font_style: 'Caption'
                        theme_text_color: 'Secondary'
                        halign: 'right'
                        size_hint_x: None
                        width: dp(100)
                
                # Beautiful empty state
                MDCard:
                    id: empty_state_card
                    orientation: 'vertical'
                    padding: dp(60)
                    spacing: dp(24)
                    md_bg_color: rgba('#F8F9FF')
                    elevation: 4
                    radius: [dp(24)]
                    opacity: 1 if (not results_rv.children and not app.loading) else 0
                    disabled: bool(results_rv.children) or app.loading

                    MDCard:
                        size_hint: None, None
                        size: dp(120), dp(120)
                        md_bg_color: rgba('#E8EAF6')
                        elevation: 4
                        radius: [dp(60)]
                        padding: dp(30)
                        pos_hint: {'center_x': 0.5}
                        
                        MDIcon:
                            icon: 'file-search-outline'
                            font_size: dp(60)
                            halign: 'center'
                            theme_text_color: 'Custom'
                            text_color: rgba('#5E35B1')
                            pos_hint: {'center_x': 0.5, 'center_y': 0.5}

                    MDLabel:
                        text: '🚀 Ready to Discover!' if app.folder_selected else '📂 Let\'s Get Started!'
                        font_style: 'H3'
                        halign: 'center'
                        theme_text_color: 'Custom'
                        text_color: rgba('#1A237E')
                        bold: True
                        size_hint_y: None
                        height: dp(48)

                    MDLabel:
                        text: 'Type your search query above and let AI find your documents instantly' if app.folder_selected else 'Click the "📁 Select Folder" button to choose your document folder'
                        font_style: 'H6'
                        halign: 'center'
                        theme_text_color: 'Secondary'
                        size_hint_y: None
                        height: dp(56)
                        text_size: self.width, None
                
                # Results showcase
                ScrollView:
                    id: results_scroll
                    bar_width: dp(12)
                    bar_color: rgba('#667EEA')
                    bar_inactive_color: rgba('#E0E0E0')
                    smooth_scroll_end: 10
                    
                    MDGridLayout:
                        id: results_rv
                        cols: 1
                        spacing: dp(24)
                        padding: [dp(8), dp(24), dp(8), dp(24)]
                        adaptive_height: True
                        size_hint_y: None
'''


from kivymd.uix.spinner import MDSpinner  # noqa: E402
from kivymd.uix.card import MDCard  # noqa: E402
from kivymd.uix.textfield import MDTextField  # noqa: E402
from kivymd.uix.progressbar import MDProgressBar  # noqa: E402
from kivymd.uix.boxlayout import MDBoxLayout  # noqa: E402
from kivymd.uix.gridlayout import MDGridLayout  # noqa: E402
from kivymd.uix.list import MDList, OneLineIconListItem, IconLeftWidget  # noqa: E402
from kivymd.uix.label import MDIcon  # noqa: E402
from kivymd.uix.selectioncontrol import MDSwitch  # noqa: E402
from kivymd.uix.fitimage import FitImage  # noqa: E402


class ResultCard(MDCard):
    file_name = StringProperty("")
    snippet = StringProperty("")
    score_percent = NumericProperty(0)
    full_path = StringProperty("")
    file_ext = StringProperty("")
    score_color = ListProperty([0.0, 0.0, 0.0, 1.0])
    
    def on_score_percent(self, instance, value):
        """Update score color when score changes"""
        try:
            s = int(float(value))
        except Exception:
            s = 0
        
        try:
            if s > 70:
                c = get_color_from_hex('#4CAF50')
            elif s > 40:
                c = get_color_from_hex('#FF9800')
            else:
                c = get_color_from_hex('#F44336')
            if c is not None and c:
                self.score_color = list(c)
            else:
                self.score_color = [0.0, 0.0, 0.0, 1.0]
        except Exception:
            self.score_color = [0.0, 0.0, 0.0, 1.0]


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
    selected_folder_name = StringProperty("")
    folder_selected = BooleanProperty(False)
    query_ready = BooleanProperty(False)
    loading = BooleanProperty(False)
    status_text = StringProperty("Ready")
    status_indicator = StringProperty("● No folder selected")
    status_indicator_color = ListProperty([0.46, 0.46, 0.46, 1.0])  # Default gray color
    indexing_progress = StringProperty("")
    indexing_progress_value = NumericProperty(0)
    settings_auto_index = BooleanProperty(False)
    settings_include_subfolders = BooleanProperty(True)
    settings_file_types = StringProperty("PDF, DOCX, TXT")
    settings_max_results = StringProperty("10")
    _last_query: str = ""
    
    def on_folder_selected(self, instance, value):
        """Update status indicator color when folder selection changes"""
        try:
            if value:
                c = get_color_from_hex('#4CAF50')
            else:
                c = get_color_from_hex('#757575')
            if c is not None and c:
                self.status_indicator_color = list(c)
            else:
                self.status_indicator_color = [0.46, 0.46, 0.46, 1.0]
        except Exception:
            self.status_indicator_color = [0.46, 0.46, 0.46, 1.0]

    def build(self):
        self.title = "DocuFindLocal - Offline AI Document Search"
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Light"
        self.root = Builder.load_string(KV)

        self._dialog: Optional[MDDialog] = None
        self._file_manager: Optional[MDFileManager] = None
        self._backend_loaded = False

        data_dir = _app_data_dir(self)
        self.db_path = data_dir / "local_index.sqlite3"
        self.cache_dir = data_dir / "model_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        _copy_bundled_cache_if_present(self.cache_dir)
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(self.cache_dir))

        # Lazy-load backend after UI renders
        Clock.schedule_once(lambda dt: self._lazy_load_backend(), 0.5)
        
        return self.root

    def _lazy_load_backend(self) -> None:
        """Load heavy backend imports after UI is visible"""
        global LocalIndexer, LocalSearcher
        
        def _load():
            global LocalIndexer, LocalSearcher
            try:
                from docufind_local.local_search.indexer import LocalIndexer as LI
                from docufind_local.local_search.searcher import LocalSearcher as LS
                LocalIndexer = LI
                LocalSearcher = LS
                Clock.schedule_once(lambda dt: self._init_backend(), 0)
            except Exception as e:
                Clock.schedule_once(lambda dt: self._show_error(f"Failed to load backend: {e}"), 0)
        
        threading.Thread(target=_load, daemon=True).start()

    def _init_backend(self) -> None:
        """Initialize backend components after lazy loading"""
        try:
            self.indexer = LocalIndexer(db_path=self.db_path, cache_dir=self.cache_dir)
            self.searcher = LocalSearcher(db_path=self.db_path, cache_dir=self.cache_dir)
            self._backend_loaded = True
            self._set_status("Ready - Backend loaded")
        except Exception as e:
            self._show_error(f"Backend initialization failed: {e}")

    def _set_status(self, text: str) -> None:
        self.status_text = text or ""
        
        # Update status indicator
        if self.loading:
            if "Indexing" in text or "Syncing" in text:
                self.status_indicator = "Indexing..."
            elif "Searching" in text:
                self.status_indicator = "Searching..."
            else:
                self.status_indicator = "Working..."
        elif self.folder_selected:
            self.status_indicator = "Ready"
        else:
            self.status_indicator = "No folder"

    def _show_error(self, text: str) -> None:
        if self._dialog:
            try:
                self._dialog.dismiss()
            except Exception:
                pass
        self._dialog = MDDialog(
            title="Error",
            text=text,
            buttons=[MDRaisedButton(text="OK", on_release=lambda *_: self._dialog.dismiss())],
        )
        self._dialog.open()

    def _toast(self, text: str) -> None:
        self._set_status(text)
        try:
            from kivymd.uix.snackbar import Snackbar
            Snackbar(
                text=text,
                snackbar_x="10dp",
                snackbar_y="10dp",
                size_hint_x=0.9,
                bg_color=self.theme_cls.primary_color
            ).open()
        except Exception:
            pass

    def show_about_dialog(self) -> None:
        if self._dialog:
            try:
                self._dialog.dismiss()
            except Exception:
                pass
        
        about_text = (
            "DocuFindLocal - Offline AI Document Search\n\n"
            "Search documents by meaning, not keywords.\n"
            "All processing happens locally on your PC.\n\n"
            "Features:\n"
            "• 100% offline - No internet required\n"
            "• Privacy-safe - Your data never leaves your device\n"
            "• AI-powered search - Understands document meaning\n"
            "• Multiple file formats - PDF, DOCX, TXT, images\n"
            "• Fast local search - Powered by vector embeddings\n\n"
            "Powered by local embeddings and vector search"
        )
        
        self._dialog = MDDialog(
            title="About DocuFindLocal",
            text=about_text,
            buttons=[MDRaisedButton(text="Close", on_release=lambda *_: self._dialog.dismiss())],
        )
        self._dialog.open()

    def show_settings_dialog(self) -> None:
        if self._dialog:
            try:
                self._dialog.dismiss()
            except Exception:
                pass

        content = Factory.SettingsContent()

        def _save_settings(*_args) -> None:
            try:
                self.settings_auto_index = bool(content.ids.auto_index_switch.active)
                self.settings_include_subfolders = bool(content.ids.include_subfolders_switch.active)
                self.settings_file_types = content.ids.filetypes_input.text or self.settings_file_types
                self.settings_max_results = content.ids.max_results_input.text or self.settings_max_results
            except Exception:
                pass
            if self._dialog:
                self._dialog.dismiss()

        self._dialog = MDDialog(
            title="Settings",
            type="custom",
            content_cls=content,
            size_hint=(0.8, None),
            buttons=[
                MDFlatButton(text="Cancel", on_release=lambda *_: self._dialog.dismiss()),
                MDRaisedButton(text="Save", on_release=_save_settings),
            ],
        )
        self._dialog.open()

    def on_query_changed(self, text: str) -> None:
        self.query_ready = bool((text or "").strip())

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
            title="Clear Index?",
            text=(
                "This will delete the local search index database.\n\n"
                "You will need to re-index your folder to search again.\n\n"
                "This action cannot be undone."
            ),
            buttons=[
                MDFlatButton(text="Cancel", on_release=_cancel),
                MDRaisedButton(text="Clear Index", on_release=_confirm, md_bg_color=(1, 0.3, 0.3, 1)),
            ],
        )
        self._dialog.open()

    def _start_clear_index(self) -> None:
        if not self._backend_loaded:
            self._show_error("Backend not loaded yet. Please wait.")
            return
            
        self.loading = True
        self.root.ids.results_rv.clear_widgets()
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

            msg = "Index cleared successfully" if deleted_any else "Index already cleared"
            Clock.schedule_once(lambda dt, _m=msg: self._on_clear_index_done(_m), 0)
        except Exception as e:
            err = str(e)
            Clock.schedule_once(lambda dt, _e=err: self._on_clear_index_error(_e), 0)

    def _on_clear_index_done(self, msg: str) -> None:
        self.loading = False
        self.root.ids.results_rv.clear_widgets()
        self.root.ids.results_count_label.text = ""
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
        self.selected_folder_name = p.name
        self.folder_selected = True
        self._set_status("Ready")
        # Clear any previous results
        self.root.ids.results_rv.clear_widgets()
        self.root.ids.results_count_label.text = ""

    def on_sync_clicked(self) -> None:
        if not self._backend_loaded:
            self._show_error("Backend not loaded yet. Please wait.")
            return
            
        if not self.folder_selected:
            self._show_error("Please select a folder first.")
            return

        if not self.indexer.embedder.available and not self.indexer.clip.available:
            self._show_error(
                "No embedding backend is available.\n\n"
                f"Text embedder error: {self.indexer.embedder.init_error}\n"
                f"Image embedder error: {self.indexer.clip.init_error}\n\n"
                "Install fastembed + onnxruntime."
            )
            return

        self.loading = True
        self.root.ids.results_rv.clear_widgets()
        self.root.ids.results_count_label.text = ""
        self.indexing_progress = "Starting indexing..."
        self.indexing_progress_value = 0
        self._set_status("Indexing folder...")
        threading.Thread(target=self._background_sync, args=(self.selected_folder,), daemon=True).start()

    def _background_sync(self, folder: str) -> None:
        try:
            def progress(msg: str) -> None:
                Clock.schedule_once(lambda dt, _m=msg: self._update_sync_progress(_m), 0)

            stats = self.indexer.index_folder(folder, progress=progress)
            final = f"Indexed {stats.indexed} documents ({stats.skipped} skipped, {stats.failed} failed)"
            Clock.schedule_once(lambda dt, _m=final: self._on_sync_done(_m), 0)
        except Exception as e:
            err = str(e)
            Clock.schedule_once(lambda dt, _e=err: self._on_sync_error(_e), 0)

    def _update_sync_progress(self, msg: str) -> None:
        """Update progress bar and message during indexing"""
        self._set_status(msg)
        self.indexing_progress = msg
        
        # Try to extract numbers from progress messages
        try:
            if "/" in msg:
                parts = msg.split("/")
                if len(parts) == 2:
                    current = int(''.join(filter(str.isdigit, parts[0])))
                    total = int(''.join(filter(str.isdigit, parts[1].split()[0])))
                    if total > 0:
                        self.indexing_progress_value = (current / total) * 100
        except Exception:
            pass

    def _on_sync_done(self, msg: str) -> None:
        self.loading = False
        self.indexing_progress = ""
        self.indexing_progress_value = 0
        self._set_status(msg)
        self._toast(msg)

    def _on_sync_error(self, err: str) -> None:
        self.loading = False
        self.indexing_progress = ""
        self.indexing_progress_value = 0
        self._show_error(f"Indexing failed: {err}")

    def on_search_clicked(self) -> None:
        if not self._backend_loaded:
            self._show_error("Backend not loaded yet. Please wait.")
            return
            
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
                f"Image embedder error: {self.searcher.clip.init_error}\n\n"
                "Install fastembed + onnxruntime."
            )
            return

        try:
            limit = int(self.root.ids.limit_input.text or 10)
        except Exception:
            limit = 10
        limit = max(1, min(50, limit))

        self._last_query = query

        self.loading = True
        self.root.ids.results_rv.clear_widgets()
        self.root.ids.results_count_label.text = ""
        self._set_status("Searching...")
        threading.Thread(target=self._background_search, args=(self.selected_folder, query, limit), daemon=True).start()

    def _background_search(self, folder: str, query: str, limit: int) -> None:
        try:
            results = self.searcher.search(folder=folder, query=query, limit=limit)
            Clock.schedule_once(lambda dt, _r=results: self._on_search_done(_r), 0)
        except Exception as e:
            err = str(e)
            Clock.schedule_once(lambda dt, _e=err: self._on_search_error(_e), 0)

    def _build_snippet(self, text: str, query: str, max_len: int = 160) -> str:
        base = (text or "").strip()
        if not base:
            return "No preview available"
        if len(base) > max_len:
            base = base[: max_len - 3] + "..."
        if query:
            try:
                pattern = re.compile(re.escape(query), re.IGNORECASE)
                base = pattern.sub(lambda m: f"[b]{m.group(0)}[/b]", base)
            except Exception:
                pass
        return base

    def icon_for_ext(self, ext: str) -> str:
        """Return a material icon name for a file extension."""
        try:
            e = (ext or "").lower()
        except Exception:
            e = ""
        mapping = {
            '.pdf': 'file-pdf-box',
            '.docx': 'file-word-box',
            '.txt': 'file-document-outline',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.pptx': 'file-powerpoint-box',
            '.xlsx': 'file-excel-box',
        }
        return mapping.get(e, 'file-outline')

    def color_for_score(self, score: Any) -> list:
        """Return an RGBA color list for a numeric score (0-100)."""
        try:
            s = int(float(score))
        except Exception:
            s = 0
        
        try:
            if s > 70:
                c = get_color_from_hex('#4CAF50')
            elif s > 40:
                c = get_color_from_hex('#FF9800')
            else:
                c = get_color_from_hex('#F44336')
            # Ensure a valid RGBA iterable is returned
            if c is None or not c:
                return [0.0, 0.0, 0.0, 1.0]
            return list(c)
        except Exception:
            return [0.0, 0.0, 0.0, 1.0]

    def color_for_folder_selected(self) -> list:
        """Return an RGBA color list for header status based on folder selection."""
        try:
            if self.folder_selected:
                c = get_color_from_hex('#4CAF50')
            else:
                c = get_color_from_hex('#757575')
            if c is None or not c:
                return [0.0, 0.0, 0.0, 1.0]
            return list(c)
        except Exception:
            return [0.0, 0.0, 0.0, 1.0]

    def _on_search_done(self, results) -> None:
        self.loading = False
        results_container = self.root.ids.results_rv
        results_container.clear_widgets()
        
        if not results:
            self._set_status("No results found. Try a different query.")
            self.root.ids.results_count_label.text = "0 results"
            return

        # Create result cards
        for r in results:
            file_path = Path(r.path)
            file_name = file_path.name
            
            # Create snippet from rel_path (no backend change) and highlight query terms
            snippet = self._build_snippet(getattr(r, "rel_path", ""), self._last_query)
            
            # Convert score to percentage
            score_percent = int(r.score * 100)
            
            # Get file extension
            file_ext = file_path.suffix.lower()
            
            card = ResultCard()
            card.file_name = file_name
            card.snippet = snippet
            card.score_percent = score_percent
            card.full_path = r.path
            card.file_ext = file_ext
            
            results_container.add_widget(card)
        
        self._set_status(f"Found {len(results)} results")
        self.root.ids.results_count_label.text = f"{len(results)} result{'s' if len(results) != 1 else ''}"

    def _on_search_error(self, err: str) -> None:
        self.loading = False
        self.root.ids.results_count_label.text = ""
        self._show_error(f"Search failed: {err}")

    def open_result(self, full_path: str) -> None:
        if not full_path:
            return
        if platform == "android":
            self._set_status("Open file is desktop-only")
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

    def open_folder_location(self, full_path: str) -> None:
        """Open the folder containing the file and highlight it"""
        if not full_path:
            return
        try:
            p = Path(full_path)
            
            if not p.exists():
                self._show_error("File not found on disk")
                return
            
            # Highlight file in folder
            if platform == "win":
                # Windows: use explorer /select to highlight the file
                subprocess.run(["explorer", f"/select,{str(p.absolute())}"], check=False)
            elif platform == "macosx":
                # macOS: use open -R to reveal in Finder
                subprocess.run(['open', '-R', str(p.absolute())], check=False)
            else:
                # Linux: open folder (most file managers don't support highlighting)
                folder = p.parent
                subprocess.run(['xdg-open', str(folder)], check=False)
        except Exception as e:
            self._show_error(f"Could not open folder: {e}")


if __name__ == "__main__":
    DocuFindLocalApp().run()