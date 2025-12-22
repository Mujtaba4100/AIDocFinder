from PyInstaller.utils.hooks import collect_data_files, collect_submodules


hiddenimports = collect_submodules("rapidocr_onnxruntime")
datas = collect_data_files("rapidocr_onnxruntime")
