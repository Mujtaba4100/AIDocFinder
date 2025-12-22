from __future__ import annotations


def main() -> None:
    # Single supported entrypoint for the local desktop app.
    from docufind_local.ui.kivy_local_app import DocuFindLocalApp

    DocuFindLocalApp().run()


if __name__ == "__main__":
    main()
