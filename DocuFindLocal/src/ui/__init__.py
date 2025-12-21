"""UI package.

Keep this package import lightweight.

Some UI entrypoints (e.g. Streamlit) have optional dependencies; importing them
at package import time breaks other entrypoints (e.g. local Kivy desktop app)
when those optional deps are not installed.
"""

__all__ = ["streamlit_app", "cli"]
