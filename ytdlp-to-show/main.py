# /!usr/bin/env python

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import dateutil
import fauxlogger as _log
import requests
import yt_dlp
import yt_dlp.options
from yt_dlp import YoutubeDL

DATA_DIR = os.getenv("DATA_DIR") or "./data"
ytdlpconf_file = os.path.join(DATA_DIR, "yt-dlp.conf")
using_ytdlpconf = os.path.exists(ytdlpconf_file)
VERBOSITY = os.getenv("YTS_VERBOSITY") or 3

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
    url: str, dest_dir: Path, override_filename: str | None = None
) -> Path:
    """Download file url to path dest_dir [optionally with filename override_filename] while keeping original extension"""
    """# download_file("https://example.com/file?id=123", Path("/home/you/Downloads"), override_filename="my_new_name")"""

    dest_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get the server-suggested filename
    filename = get_filename_from_cd(response.headers.get("content-disposition", ""))

    if not filename:
        filename = Path(url).name

    if override_filename:
        ext = Path(filename).suffix
        filename = f"{override_filename}{ext}"

    out_file = dest_dir / filename

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


create_parser = yt_dlp.options.create_parser


def parse_patched_options(opts: list):
    patched_parser = create_parser()
    patched_parser.defaults.update(
        {
            "ignoreerrors": False,
        }
    )
    yt_dlp.options.create_parser = lambda: patched_parser
    try:
        return yt_dlp.parse_options(opts)
    finally:
        yt_dlp.options.create_parser = create_parser


default_ytdlp_opts = dict(parse_patched_options([]).ydl_opts)


def cli_to_api(opts: list, cli_defaults: bool = False) -> dict:
    new_opts = (yt_dlp.parse_options if cli_defaults else parse_patched_options)(
        opts
    ).ydl_opts

    diff: dict = {k: v for k, v in new_opts.items() if default_ytdlp_opts[k] != v}
    if "postprocessors" in diff:
        diff["postprocessors"] = [
            pp
            for pp in diff["postprocessors"]
            if pp not in default_ytdlp_opts["postprocessors"]
        ]
    return diff


def sanitize(name: str) -> str:
    _str = re.sub(r'[\\/:*?"<>|]', "", name)
    _str = re.sub(r"\s+", " ", _str).strip()
    _str = re.sub(r"\W+$", "", _str)
    return _str


def read_url_from_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().strip()


def should_create_episode(video_info: dict) -> bool:
    """
    Stub hook for future logic.
    Return False to skip creating episode files for a video.
    """
    if video_info.get("duration", 181) < 180:
        return False  # Shorts

    return True


def duration_filter(info, *, incomplete):
    if incomplete:
        return None

    duration = info.get("duration")
    if duration is None:
        return None

    if duration < 181:
        return f"Skipping (duration {duration}s < {181}s)"

    return None


# ----------------------------
# Step 1: Download metadata
# ----------------------------


def download_channel(url: str, output_root: Path) -> Path:
    ydl_opts = {}

    if using_ytdlpconf:
        ydl_opts.update(cli_to_api(["--config-locations", f"{DATA_DIR}"]))

    ydl_opts.update(
        {
            "match_filter": duration_filter,
            "skip_download": False,
            "restrictfilenames": True,
            "extract_flat": False,
            "writeinfojson": True,
            "writethumbnail": True,
            "writeplaylistmetafiles": True,
            "outtmpl": str(output_root / "%(channel_id)s" / "%(id)s" / "video.%(ext)s"),
            "quiet": False,
            "download_archive": str(output_root / "download_archive.lst"),
            "sleep_interval": 90,
            "max_sleep_interval": 180,
            "sleep_interval_requests": 2,
            "concurrent_fragment_downloads": 2,
            # "playlist_items": "1-3",
        }
    )

    _log.msg(f"Downloading videos from {url} to {output_root} ")

    with YoutubeDL(ydl_opts) as ydl:  # pyright: ignore[reportArgumentType]
        info = ydl.extract_info(url, download=True)

    _log.msg("Download finished.")

    channel_dir = output_root / (info.get("channel_id") or "")
    # channel_dir.mkdir(parents=True, exist_ok=True)

    return channel_dir


# ----------------------------
# Step 2: TV show NFO
# ----------------------------


def download_show_images(image_data: dict, show_dir: Path):
    _log.msg(f"Downloading images to {show_dir}")

    for _entry in image_data:
        # id, url, pref, height, width
        _entry["pixel_count"] = _entry.get("height", 1) * _entry.get("width", 1)

    landscape_idx = {
        _key: _val
        for _key, _val in enumerate(
            sorted(
                (
                    _thumb
                    for _thumb in image_data
                    if isinstance(_thumb["id"], int) or str_is_int(_thumb["id"])
                    if _thumb.get("width", 1) > _thumb.get("height", 1)
                ),
                key=lambda _entry: (_entry.get("preference", 0), _entry["pixel_count"]),
            )
        )
    }
    _log.msg(f"Landscapes: {len(landscape_idx)}")
    for _id, _entry in landscape_idx.items():
        download_file(
            _entry.get("url", ""),
            show_dir,
            f"landscape{_id}" if _id > 0 else "landscape",
        )

    poster_idx = {
        _key: _val
        for _key, _val in enumerate(
            sorted(
                (
                    _thumb
                    for _thumb in image_data
                    if isinstance(_thumb["id"], int) or str_is_int(_thumb["id"])
                    if _thumb.get("width", 1) < _thumb.get("height", 1)
                ),
                key=lambda _entry: (_entry.get("preference", 0), _entry["pixel_count"]),
            )
        )
    }
    _log.msg(f"Posters: {len(poster_idx)}")
    for _id, _entry in poster_idx.items():
        download_file(
            _entry.get("url", ""), show_dir, f"poster{_id}" if _id > 0 else "poster"
        )

    fanart_idx = {
        _key: _val
        for _key, _val in enumerate(
            sorted(
                (
                    _thumb
                    for _thumb in image_data
                    if isinstance(_thumb["id"], int) or str_is_int(_thumb["id"])
                    if _thumb.get("width", 0) == _thumb.get("height", 0)
                ),
                key=lambda _entry: (_entry.get("preference", 0), _entry["pixel_count"]),
            )
        )
    }
    _log.msg(f"Fanart: {len(fanart_idx)}")
    for _id, _entry in fanart_idx.items():
        download_file(
            _entry.get("url", ""), show_dir, f"fanart{_id}" if _id > 0 else "fanart"
        )

    named_art = {
        _thumb["id"]: _thumb
        for _thumb in image_data
        if not (isinstance(_thumb["id"], int) or str_is_int(_thumb["id"]))
    }
    _log.msg(f"Named images: {len(named_art)}")
    for _id, _entry in named_art.items():
        download_file(_entry.get("url", ""), show_dir, _id.replace("_uncropped", ""))


def create_tvshow_nfo(channel_dir: Path, library_root: Path) -> tuple[Path, str]:
    info_dir = channel_dir / channel_dir.name
    info_file = info_dir / "video.info.json"
    first_info = json.load(open(info_file, encoding="utf-8"))

    show_name = sanitize(first_info.get("channel"))
    description = first_info.get("description", "")
    u_id = first_info.get("id")
    u_id_type = first_info.get("webpage_url_domain")
    show_dir = library_root / show_name
    show_dir.mkdir(parents=True, exist_ok=True)
    image_data = first_info.get("thumbnails", [])

    download_show_images(image_data, show_dir)

    tvshow_nfo = show_dir / "tvshow.nfo"
    tvshow_nfo.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<tvshow>
  <title>{show_name}</title>
  <plot>{description}</plot>
  <uniqueid type="{u_id_type}" default="">{u_id}</uniqueid>
  <studio>{u_id_type}</studio>
</tvshow>
""",
        encoding="utf-8",
    )

    for _img in info_dir.iterdir():
        if _img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            _dest = (show_dir / show_name).with_suffix(_img.suffix)
            shutil.copy(_img, _dest)

    return show_dir, show_name


# ----------------------------
# Step 3: Episode NFOs
# ----------------------------


def create_episode_nfos(channel_dir: Path, show_dir: Path):
    episode_folders = sorted(
        _dir
        for _dir in channel_dir.iterdir()
        if _dir.is_dir() and _dir.name != channel_dir.name
    )

    # Group videos by year
    videos_by_year = defaultdict(list)

    for _ep_path in episode_folders:
        _info = json.load(open(_ep_path / "video.info.json", encoding="utf-8"))

        if not should_create_episode(_info):
            continue

        upload_date = dateutil.parser.parse(_info.get("upload_date"))
        if not upload_date:
            continue

        year = upload_date.year

        videos_by_year[year].append((upload_date, _ep_path, _info))

    for year, videos in videos_by_year.items():
        # Sort by date ascending within year
        season_dir = show_dir / f"Season {year}"
        season_dir.mkdir(parents=True, exist_ok=True)

        videos.sort(key=lambda x: x[0])

        for episode_num, (date, _ep_path, _info) in enumerate(videos, start=1):
            u_id = _info.get("id")
            u_id_type = _info.get("webpage_url_domain")
            title = _info.get("title")
            description = _info.get("description", "")
            aired = date.date().isoformat()

            safe_title = sanitize(title)
            base_name = f"{show_dir.name} - S{year}E{episode_num:02d} - {safe_title}"

            # Episode NFO
            nfo_path = season_dir / f"{base_name}.nfo"
            nfo_path.write_text(
                f"""<?xml version="1.0" encoding="UTF-8"?>
<episodedetails>
  <title>{title}</title>
  <season>{year}</season>
  <episode>{episode_num}</episode>
  <aired>{aired}</aired>
  <plot>{description}</plot>
  <uniqueid type="{u_id_type}" default="">{u_id}</uniqueid>
  <studio>{u_id_type}</studio>
</episodedetails>
""",
                encoding="utf-8",
            )

            # Episode thumbnail
            thumbnail_url = _info.get("thumbnail", "")
            if thumbnail_url:
                download_file(thumbnail_url, season_dir, f"{base_name}-thumb")

            for _img in _ep_path.iterdir():
                if _img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    _dest = (season_dir / base_name).with_suffix(_img.suffix)
                    if not _dest.exists:
                        try:
                            os.link(_img, _dest)
                        except OSError:
                            shutil.copy2(_img, _dest)

            if not (season_dir / f"{base_name}.mkv").exists():
                try:
                    os.link(
                        _ep_path / "video.mkv",
                        season_dir / f"{base_name}.mkv",
                    )
                except OSError:
                    shutil.copy2(
                        _ep_path / "video.mkv",
                        season_dir / f"{base_name}.mkv",
                    )


# ----------------------------
# Main
# ----------------------------


def main():
    # Verbosity 0
    import argparse

    from fauxjson import load_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--url-file", required=True)
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--library-dir", required=True)

    args = parser.parse_args()

    urls = load_json(args.url_file)
    download_dir = Path(args.download_dir)
    library_dir = Path(args.library_dir)

    if not urls:
        return

    for url in urls:
        _log.msg(f"Processing next URL: {url}")
        channel_dir = download_channel(url, download_dir)
        _log.msg(f"Channel stored at {channel_dir}")
        show_dir, show_name = create_tvshow_nfo(channel_dir, library_dir)
        _log.msg(f"Show data for {show_name} stored to library at {show_dir}")
        create_episode_nfos(channel_dir, show_dir)
        _log.msg(f"Episodes and data stored to library in {show_dir}")

        _log.msg(f"Done with {channel_dir}.")


if __name__ == "__main__":
    main()
