from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# fastembed performs dynamic imports and may include model/data files.
hiddenimports = collect_submodules('fastembed') or []
datas = collect_data_files('fastembed') or []
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules("fastembed")
datas = collect_data_files("fastembed")
