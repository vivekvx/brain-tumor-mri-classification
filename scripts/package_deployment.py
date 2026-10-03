"""Package the selected Streamlit model and app for hosting."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    metadata = json.loads(Path("models/deployment.json").read_text())
    model = Path("models") / metadata["model_path"]
    if not model.is_file():
        raise FileNotFoundError(model)
    destination = Path("deployment/brain_mri_streamlit.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.write("app.py")
        bundle.write("requirements-deploy.txt", "requirements.txt")
        bundle.write("models/deployment.json")
        bundle.write(model)
        bundle.write("src/__init__.py")
        bundle.write("src/inference.py")
        bundle.writestr("README.txt", "Install requirements with pip install -r requirements.txt, then run streamlit run app.py. Research demonstration only; not for clinical use.\n")
    print(destination, f"({destination.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
