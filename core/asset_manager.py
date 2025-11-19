import os
import urllib.request
from pathlib import Path

DEFAULT_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
)


def _download_file(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(destination, "wb") as out_file:
        out_file.write(response.read())


def ensure_eye_cascade(path: str, url: str = None, force: bool = False) -> Path:
    """Ensure the Haar cascade used by IrisDetector is available locally."""
    destination = Path(path)
    if destination.exists() and not force:
        return destination

    download_url = url or os.environ.get("IRIS_EYE_CASCADE_URL", DEFAULT_CASCADE_URL)
    if not download_url:
        raise FileNotFoundError(
            "Eye cascade file is missing and no download URL was provided via "
            "IRIS_EYE_CASCADE_URL."
        )

    try:
        _download_file(download_url, destination)
    except Exception as exc:  # pragma: no cover - network/IO errors
        raise FileNotFoundError(
            f"Could not download eye cascade from {download_url}: {exc}"
        ) from exc
    return destination
