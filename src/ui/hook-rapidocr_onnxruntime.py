from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# rapidocr_onnxruntime may dynamically load models and native libs.
hiddenimports = collect_submodules('rapidocr_onnxruntime') or []
datas = collect_data_files('rapidocr_onnxruntime') or []
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules("rapidocr_onnxruntime")
datas = collect_data_files("rapidocr_onnxruntime")
