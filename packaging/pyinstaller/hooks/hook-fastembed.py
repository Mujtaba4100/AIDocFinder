from PyInstaller.utils.hooks import collect_data_files, collect_submodules


hiddenimports = collect_submodules("fastembed")
datas = collect_data_files("fastembed")
