from PyInstaller.utils.hooks import collect_data_files, collect_submodules


# KivyMD uses dynamic imports for icon definitions; collect everything.
hiddenimports = collect_submodules("kivymd")
datas = collect_data_files("kivymd")

if "kivymd.icon_definitions.md_icons" not in hiddenimports:
    hiddenimports.append("kivymd.icon_definitions.md_icons")
