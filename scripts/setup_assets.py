import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.asset_manager import DEFAULT_CASCADE_URL, ensure_eye_cascade


def main():
    parser = argparse.ArgumentParser(description="Download placeholder model assets.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the cascade already exists.",
    )
    parser.add_argument(
        "--cascade-url",
        default=DEFAULT_CASCADE_URL,
        help="Override the download URL for the OpenCV eye cascade.",
    )
    args = parser.parse_args()

    cascade_path = Path("data/haarcascade_eye.xml")
    cascade_file = ensure_eye_cascade(
        str(cascade_path),
        url=args.cascade_url,
        force=args.force,
    )
    print(f"Eye cascade ready at {cascade_file}")

    models_root = Path("models")
    for sub in ("segmentation", "encoding"):
        target = models_root / sub
        target.mkdir(parents=True, exist_ok=True)
    print(
        "Created model directories at models/segmentation and models/encoding. "
        "Place your RITnet and DeepIrisNet2 weights there when available."
    )


if __name__ == "__main__":
    main()
