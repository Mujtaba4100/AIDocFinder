# hook-kivymd.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# include all submodules and data files of kivymd (icons, kv, fonts, etc.)
hiddenimports = collect_submodules("kivymd")
datas = collect_data_files("kivymd")