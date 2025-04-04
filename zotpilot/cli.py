import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).parent.parent / "app.py"
    args = ["streamlit", "run", str(app_path)] + sys.argv[1:]
    subprocess.run(args, check=False)
