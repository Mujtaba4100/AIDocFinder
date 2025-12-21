from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all submodules and data files from kivymd. This ensures dynamic
# icon/font/resource modules like kivymd.icon_definitions.md_icons are
# bundled into the onefile exe.
hiddenimports = collect_submodules('kivymd') or []
datas = collect_data_files('kivymd') or []

# Some environments may reference these directly; ensure icon definitions
# are explicitly present.
if 'kivymd.icon_definitions.md_icons' not in hiddenimports:
    hiddenimports.append('kivymd.icon_definitions.md_icons')
# hook-kivymd.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# include all submodules and data files of kivymd (icons, kv, fonts, etc.)
hiddenimports = collect_submodules("kivymd")
datas = collect_data_files("kivymd")