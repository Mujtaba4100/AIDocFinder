import os
import sys

# Ensure the project root (where the `src` folder lives) is on sys.path
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Now import and run the local Kivy app using the same module path used in development
from src.ui.kivy_local_app import DocuFindLocalApp

if __name__ == "__main__":
    DocuFindLocalApp().run()
