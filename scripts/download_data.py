from pathlib import Path
import os
import zipfile
import subprocess
import shutil

from dotenv import load_dotenv


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        raise RuntimeError(
            "Missing KAGGLE_USERNAME or KAGGLE_KEY. "
            "Create a .env file in the project root using .env.example."
        )

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found in your environment. "
            "Install it with: pip install kaggle"
        )

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key

    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "earnings-call-transcripts.zip"
    extract_dir = raw_dir / "earnings-call-transcripts"
    extract_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "ashwinm500/earnings-call-transcripts",
        "-p",
        str(raw_dir),
        "--force",
    ]

    subprocess.run(cmd, check=True, env=env)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Expected downloaded file not found: {zip_path}"
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    print(f"Downloaded zip to: {zip_path}")
    print(f"Extracted dataset to: {extract_dir}")


if __name__ == "__main__":
    main()