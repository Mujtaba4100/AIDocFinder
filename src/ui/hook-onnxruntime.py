from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Ensure onnxruntime python package and data files are collected. The
# native DLLs often live next to the package; PyInstaller will detect them
# but collecting data helps in some setups.
hiddenimports = collect_submodules('onnxruntime') or []
datas = collect_data_files('onnxruntime') or []
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

hiddenimports = collect_submodules("onnxruntime")
binaries = collect_dynamic_libs("onnxruntime")
