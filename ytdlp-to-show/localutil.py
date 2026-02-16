import regex as re
from pathlib import Path
import requests


# ----------------------------
# Utility helpers
# ----------------------------


def get_filename_from_cd(cd: str) -> str | None:
    """Extract filename from Content-Disposition header."""
    if not cd:
        return None
    fname_match = re.findall('filename="?([^"]+)"?', cd)
    if fname_match:
        return fname_match[0]
    return None


def download_file(
    url: str,
    dest_dir: Path,
    override_filename: str | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download file url to path dest_dir [optionally with filename override_filename] while keeping original extension"""
    """# download_file("https://example.com/file?id=123", Path("/home/you/Downloads"), override_filename="my_new_name")"""

    dest_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    if response.status_code == 404:
        return None

    response.raise_for_status()

    # Get the server-suggested filename
    filename = get_filename_from_cd(response.headers.get("content-disposition", ""))

    if not filename:
        filename = Path(url).name

    if override_filename:
        ext = Path(filename).suffix.split("?")[0]
        filename = f"{override_filename}{ext}"

    out_file = dest_dir / filename

    if out_file.exists() and not overwrite:
        return None

    # Download the file in chunks
    with open(out_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return out_file


def str_is_int(str):
    try:
        int(str)
        return True
    except ValueError:
        return False


def sanitize(name: str) -> str:
    _str = re.sub(r'[\[\]\\/:*?"<>|]', "", name)
    _str = re.sub(r"\s+", " ", _str).strip()
    _str = re.sub(r"\W+$", "", _str)
    return _str
