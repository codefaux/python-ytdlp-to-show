# /!usr/bin/env python

import itertools
import json
import os
import re
import shutil
import time
from pathlib import Path

import dateutil
from unidecode import unidecode
import fauxlogger as _log
import requests
import yt_dlp
import yt_dlp.options
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError
from typing import Final

DATA_DIR = os.getenv("DATA_DIR") or "./data"
ytdlpconf_file = os.path.join(DATA_DIR, "yt-dlp.conf")
cookies_file = os.path.join(DATA_DIR, "cookies.txt")
using_ytdlpconf = os.path.exists(ytdlpconf_file)
using_cookies = os.path.exists(cookies_file)
VERBOSITY = os.getenv("YTS_VERBOSITY") or 3
ytdlp_options: Final[dict] = {}

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
    _str = re.sub(r'[\[\]\\/:*?"<>|]', "", name)
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


def chain_filters(*filters):
    def _filter(info, *, incomplete):
        for f in filters:
            reason = f(info, incomplete=incomplete)
            if reason:
                return reason
        return None

    return _filter


def duration_filter(info, *, incomplete):
    if incomplete:
        return None

    duration = info.get("duration")
    if duration is None:
        return None

    if duration < 181:
        return f"Skipping (duration {duration}s < {181}s)"

    return None


def live_filter(info, *, incomplete):
    if incomplete:
        return None

    if info.get("is_live") or info.get("was_live"):
        return "Skipping live or livestream VOD"

    return None


# ----------------------------
# Step 1: Download metadata
# ----------------------------


def load_archive(ytdlp_data_path: Path) -> dict[str, list[str]] | None:
    archive_file = ytdlp_data_path / "download_archive.lst"

    if not archive_file.exists():
        return None

    entries: dict[str, list[str]] = {}

    for line in archive_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        extractor, video_id = line.split(" ", 1)

        entries.setdefault(extractor, []).append(video_id)

    return entries


dl_last_bytes = 0
dl_last_change = time.time()


def anti_stall(info):

    global dl_last_bytes, dl_last_change

    if info["status"] == "downloading":
        downloaded = info.get("downloaded_bytes", 0)

        if downloaded != dl_last_bytes:
            dl_last_bytes = downloaded
            dl_last_change = time.time()
        elif time.time() - dl_last_change > 20:
            raise DownloadError("Anti-Stall: No progress detected")


def setup_ytdlp(
    output_root: Path,
    skip_download: bool = False,
    extract_flat: bool = False,
):
    ytdlp_options.clear()

    if using_ytdlpconf:
        ytdlp_options.update(cli_to_api(["--config-locations", DATA_DIR]))
    if using_ytdlpconf:
        ytdlp_options.update({"cookiefile": cookies_file})

    ytdlp_options.update(
        {
            "socket_timeout": 23,
            "retries": 2,
            "fragment_retries": 2,
            "throttled_rate": 102400,
            "progress_hooks": [anti_stall],
            "match_filter": chain_filters(duration_filter, live_filter),
            "skip_download": skip_download,
            "restrictfilenames": True,
            "extract_flat": extract_flat,
            "writeinfojson": True,
            "writethumbnail": True,
            "writeplaylistmetafiles": True,
            "outtmpl": str(output_root / "%(channel_id)s" / "%(id)s" / "video.%(ext)s"),
            "quiet": False,
            "sleep_interval": 1 if skip_download else 90,
            "max_sleep_interval": 1 if skip_download else 180,
            "sleep_interval_requests": 0 if skip_download else 1,
            "concurrent_fragment_downloads": 1 if skip_download else 2,
            "continuedl": True,
            "nopart": False,
            "subtitleslangs": ["en", "-live_chat"],
            # "playlist_items": "1-3",
        }
    )
    if skip_download:
        ytdlp_options.pop("download_archive", None)
    else:
        ytdlp_options.update(
            {"download_archive": str(output_root / "download_archive.lst")}
        )


def find_playlist_srcdir(
    source_dir: Path | None, channel_name, playlist_name
) -> Path | None:
    from cfsonarrmatcher import match_to_show

    playlist_srcdir = None

    if source_dir:
        candidate_names = list(
            (_dir.name, i)
            for i, _dir in enumerate(source_dir.iterdir())
            if _dir.is_dir()
        )

        channel_result = match_to_show(channel_name, candidate_names)
        playlist_result = match_to_show(playlist_name, candidate_names)

        results = [channel_result, playlist_result]

        best_result = max(results, key=lambda _result: _result.get("score", 0))

        channel_srcname = (
            next(
                (
                    item[0]
                    for item in candidate_names
                    if item[1] == best_result.get("matched_id")
                ),
                None,
            )
            if best_result.get("score", 0) > 100
            else None
        )

        playlist_srcdir = (
            source_dir / Path(channel_srcname) if channel_srcname else None
        )

        if playlist_srcdir and (playlist_srcdir / Path("Videos")).is_dir(
            follow_symlinks=True
        ):
            playlist_srcdir = playlist_srcdir / Path("Videos")

    return playlist_srcdir


def write_playlist_json(playlist_dir: Path, url: str, playlist_info) -> list[dict]:
    channel_name: str = playlist_info.get("channel") or ""
    playlist_name: str = playlist_info.get("title") or channel_name
    channel_id: str = playlist_info.get("channel_id") or ""

    playlist_data = [
        dict(
            {
                "index": -1,
                "title": playlist_name,
                "url": url,
                "channel_id": channel_id,
                "id": playlist_info.get("id") or "",
                "channel_name": channel_name,
            }
        )
    ]

    # --- TODO: ONLY SUPPORTS PLAYLIST VIDEOS FROM SAME CHANNEL, MUST DL EACH INFO IN SEQUENCE, FILL playlist_data
    for idx, video in enumerate(playlist_info.get("entries") or [], start=1):
        playlist_data.append(
            {
                "index": idx,
                "title": video.get("title"),
                "url": video.get("url"),
                "id": video.get("id"),
            }
        )

    (playlist_dir / Path("playlist.json")).write_text(
        json.dumps(playlist_data, indent=2), encoding="utf-8"
    )
    return playlist_data


def prune_download_urls(output_root: Path, playlist_info) -> list[tuple[str, str]]:
    download_archive = load_archive(output_root)

    urls_to_download: list[tuple[str, str]] = []

    for _entry in playlist_info.get("entries") or []:
        if not _entry:
            continue

        _ytdlp_output_file = Path(
            output_root
            / Path(playlist_info.get("channel_id") or "")
            / Path(_entry.get("id") or "")
            # / "video.mkv"
            / "video.info.json"
        )

        if _ytdlp_output_file.exists():
            continue

        if download_archive:
            if (
                _entry.get("id")
                in download_archive[playlist_info.get("extractor", "").split(":")[0]]
            ):
                continue

        urls_to_download.append((_entry.get("url") or "", _entry.get("id") or ""))
    return urls_to_download


def download_item_info(output_root, _url_tuple):
    _url = _url_tuple[0]
    _id = _url_tuple[1]
    setup_ytdlp(
        output_root, skip_download=True, extract_flat=True
    )  # Switched extract_flat to True to prevent playlist of playlists downloading each video info
    single_info = ydl_safe_extract_info(
        output_root,
        _url,
        download=True,
    )

    if single_info:
        if not isinstance(single_info, int) and chain_filters(
            duration_filter, live_filter
        )(single_info, incomplete=False):
            _extractor = single_info.get("extractor") or ""
            _log.msg(f"Adding {_extractor} {_id} to download_archive")
            add_to_archive(_extractor, _id, output_root)

    # --- TODO: ONLY SUPPORTS PLAYLIST VIDEOS FROM SAME CHANNEL, MUST DL EACH INFO IN SEQUENCE, FILL playlist_data

    # for _entry in playlist_data:
    #     if _entry.get("id") == single_info.get("id"):
    #         _entry["epoch"] = single_info.get("epoch") or 0
    #         break


def download_playlist(
    url: str, output_root: Path, source_dir: Path | None = None
) -> tuple[Path, Path | None, list[dict]]:

    # --- TODO: ONLY SUPPORTS PLAYLIST VIDEOS FROM SAME CHANNEL, MUST DL EACH INFO IN SEQUENCE, FILL playlist_data

    _log.msg(f"Downloading playlist info from {url} to {output_root} ")

    setup_ytdlp(output_root, skip_download=True, extract_flat=True)
    playlist_info = ydl_safe_extract_info(output_root, url, download=True)

    # --- TODO: Store playlist, check if changed before doing more

    if not playlist_info or isinstance(playlist_info, int):
        raise RuntimeError(f"ydl_safe_extract_info exception {playlist_info}")

    playlist_dir: Path = (
        output_root
        / Path(playlist_info.get("channel_id") or "")
        / Path(playlist_info.get("id") or "")
    )

    playlist_data = write_playlist_json(playlist_dir, url, playlist_info)

    channel_name: str = playlist_info.get("channel") or ""
    playlist_name: str = playlist_info.get("title") or channel_name
    playlist_srcdir = find_playlist_srcdir(source_dir, channel_name, playlist_name)

    urls_to_download = prune_download_urls(output_root, playlist_info)

    _tot = len(urls_to_download)
    for _i, _url_tuple in enumerate(urls_to_download, start=1):
        _log.msg(
            f"Downloading info for item {_log._GREEN}{_i}{_log._RESET} of {_log._BLUE}{_tot}{_log._RESET}: {_log._YELLOW}{_url_tuple[0]}{_log._RESET} "
        )
        download_item_info(output_root, _url_tuple)

    _log.msg("Download finished.")

    return playlist_dir, playlist_srcdir, playlist_data


def ydl_safe_extract_info(output_root: Path, *args, **kwargs):
    with YoutubeDL(ytdlp_options) as ydl:  # pyright: ignore[reportArgumentType]
        try:
            return ydl.extract_info(*args, **kwargs)
        except DownloadError as e:
            if e.msg and "members-only" in e.msg:
                _log.msg("Download skipped: Members only")
                _exc: ExtractorError = e.exc_info[
                    1
                ]  # pyright: ignore[reportAssignmentType]
                _extractor = str(_exc.ie)
                _id = _exc.video_id
                _log.msg(f"Adding {_extractor} {_id} to download_archive")
                add_to_archive(_extractor, _id, output_root)

                return 0
            elif e.msg and ("not available" in e.msg or "unavailable" in e.msg):
                _log.msg("Video unavailable, skipping")
                _exc: ExtractorError = e.exc_info[
                    1
                ]  # pyright: ignore[reportAssignmentType]
                _extractor = str(_exc.ie)
                _id = _exc.video_id
                _log.msg(f"Adding {_extractor} {_id} to download_archive")
                add_to_archive(_extractor, _id, output_root)

                return 0
            elif e.msg and "403" in e.msg.lower() and "forbidden" in e.msg.lower():
                _log.msg(f"Download error: {e.msg}")
                raise RuntimeError(e.msg)
                return 403
            elif e.msg and "bot" in e.msg.lower() and "sign in" in e.msg.lower():
                _log.msg(f"Download error: {e.msg}")
                raise RuntimeError(e.msg)
                return 403
            else:
                _log.msg(f"Download error: {e.msg}")
                _log.msg("Pausing for 30 sec before retry.")
                time.sleep(30)
    return int(-1)


def download_episode(output_root: Path, episode_info: dict) -> Path | int:
    _url = episode_info.get("webpage_url") or episode_info.get("url")
    if _url:
        setup_ytdlp(output_root, skip_download=False, extract_flat=False)
        _attempt = 0
        _max_attempt = 2

        while _attempt < _max_attempt:
            _attempt += 1

            _info = ydl_safe_extract_info(output_root, _url)

            if isinstance(_info, int):
                return _info

            _ytdlp_output_file = Path(
                output_root
                / Path(_info.get("channel_id") or "")
                / Path(_info.get("id") or "")
                / "video.mkv"
            )

            _log.msg("Episode download finished.")

            return _ytdlp_output_file

        _log.msg("Max retries.")
        raise RuntimeError("Max download attempts exceeded")
        return -1
    else:
        _log.msg("Episode download error; URL not present.")

    return -1


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


def create_tvshow_nfo(playlist_dir: Path, library_root: Path) -> Path:
    info_file = playlist_dir / "video.info.json"
    first_info = json.load(open(info_file, encoding="utf-8"))
    if playlist_dir.name == playlist_dir.parent.name:
        # --- FULL CHANNEL
        playlist_name = first_info.get("channel") or ""
        safe_playlist_name = sanitize(playlist_name)
    else:
        # --- PLAYLIST
        playlist_name = f"{first_info.get("channel")}: {first_info.get("title")}"
        safe_playlist_name = sanitize(
            f"{first_info.get("channel")} - {first_info.get("title")}"
        )
    description = first_info.get("description", "")
    u_id = first_info.get("id")
    u_id_type = first_info.get("webpage_url_domain")
    show_dir = library_root / safe_playlist_name
    show_dir.mkdir(parents=True, exist_ok=True)

    tvshow_nfo = show_dir / "tvshow.nfo"
    if not tvshow_nfo.exists():
        tvshow_nfo.write_text(
            f"""<?xml version="1.0" encoding="UTF-8"?>
<tvshow>
  <title>{playlist_name}</title>
  <plot>{description}</plot>
  <uniqueid type="{u_id_type}" default="">{u_id}</uniqueid>
  <studio>{u_id_type}</studio>
</tvshow>
""",
            encoding="utf-8",
        )

        image_data = first_info.get("thumbnails", [])
        download_show_images(image_data, show_dir)

        for _img in playlist_dir.iterdir():
            if _img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                _dest = (show_dir / safe_playlist_name).with_suffix(_img.suffix)
                _log.msg(f"Copying image {_img}")
                try:
                    os.link(_img, _dest)
                except OSError:
                    shutil.copy2(_img, _dest)

    return show_dir


# ----------------------------
# Step 3: Episode NFOs
# ----------------------------


def sort_videos_by_year(
    channel_dir: Path, abort_unmatched_year: bool = True
) -> tuple[dict, int]:
    _vby = {}
    _tot = 0

    for _ep_path in sorted(
        _dir
        for _dir in channel_dir.iterdir()
        if not (_dir / Path("playlist.json")).exists()
    ):
        _info = json.load(open(_ep_path / "video.info.json", encoding="utf-8"))

        if _info.get("_type") == "playlist":
            continue

        if not should_create_episode(_info):
            continue

        _upload_date = dateutil.parser.parse(_info.get("upload_date"))

        if not _upload_date and abort_unmatched_year:
            continue
        else:
            _year = 0

        _year = _upload_date.year

        _tot += 1
        _vby.setdefault(_year, []).append((_upload_date, _ep_path, _info))
    return _vby, _tot


def get_new_title(video_id: str) -> str | None:
    _stub = True

    if _stub:
        return None

    _response = requests.get(
        f"https://sponsor.ajay.app/api/branding?videoID={video_id}"
    )

    _response.raise_for_status()

    api_data = _response.json()

    if "titles" not in api_data or not api_data["titles"]:
        _log.msg("No new title found in the API response.")
        return None

    titles = [title.get("title") for title in api_data["titles"] if title.get("title")]

    if len(titles) <= 1:
        new_title = titles[0]
    else:
        _log.msg("Multiple titles found.")
        for i, title in titles:
            _log.msg(f"{i + 1}: {title}")

        new_title = titles[0]

    return new_title


def build_candidates(source_dir: Path) -> list[dict]:
    _series = source_dir.name
    _titles = {}

    for _file in sorted(source_dir.iterdir()):
        if _file.is_file() and _file.suffix.lower() in {".mkv", ".mp4"}:
            if _file.stem not in _titles:
                _titles[_file.stem] = {
                    "series": _series.replace("_", " "),
                    "title": _file.stem.replace("_", " "),
                    "local_filename": _file.name,
                }

    return list(_titles.values())


def find_match(video_info: dict, source_dir: Path) -> Path | None:
    from cfsonarrmatcher import match_to_episode

    # Prep data and throw to python-cfsonarr-matcher
    # build candidate dict from source_dir
    cand_dict = build_candidates(source_dir)

    output = match_to_episode(video_info.get("title") or "", "", cand_dict)

    # _log.msg(output)
    if (output.get("score") or 0) > 60:
        return output.get("full_match", {}).get("local_filename", None)

    return None


def add_to_archive(_extractor: str, _id: str, ytdlp_data_path: Path) -> None:
    _entry = f"{_extractor} {_id}"
    _file = ytdlp_data_path / "download_archive.lst"

    _file.parent.mkdir(parents=True, exist_ok=True)

    _lines = _file.read_text(encoding="utf-8").splitlines() if _file.exists() else []

    if _entry in _lines:
        return

    _lines.append(_entry)
    _file.write_text("\n".join(_lines) + "\n", encoding="utf-8")


def write_ep_nfo(
    ep_dir,
    base_name,
    title,
    orig_title,
    season_num,
    episode_num,
    airdate,
    description,
    u_id_type,
    u_id,
):
    nfo_path = ep_dir / f"{base_name}.nfo"
    nfo_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<episodedetails>
  <title>{title}</title>
  <originaltitle>{orig_title}</originaltitle>
  <season>{season_num}</season>
  <episode>{episode_num}</episode>
  <aired>{airdate}</aired>
  <plot>{description}</plot>
  <uniqueid type="{u_id_type}" default="">{u_id}</uniqueid>
  <studio>{u_id_type}</studio>
</episodedetails>
""",
        encoding="utf-8",
    )


def backfill_file(
    _entry: dict, source_dir: Path | None, _target_file: Path, _ytdlp_ep_path: Path
) -> Path:
    _match = find_match(_entry, source_dir) if source_dir else None

    _ytdlp_file = _ytdlp_ep_path / Path("video.mkv")

    if not _ytdlp_file.exists():
        _lfile = None
        if source_dir and _match:
            _lfile = source_dir / Path(_match)
        elif _target_file.exists():
            _lfile = _target_file

        if _lfile:
            try:
                os.link(_lfile, _ytdlp_file)
                _log.msg(
                    f"Hardlinked source-dir video to ytdlp-output\n\tSRC: {_lfile}\n\tDEST: {_ytdlp_file}"
                )
            except OSError:
                shutil.move(_lfile, _ytdlp_file)
                _log.msg(
                    f"Moved source-dir video to ytdlp-output\n\tSRC: {_lfile}\n\tDEST: {_ytdlp_file}"
                )

    return _ytdlp_file


def process_season_videos(
    videos: list,
    season_num: int,
    library_season_dir: Path,
    source_dir: Path | None,
    ytdlp_dir: Path,
):
    total_videos = len(videos)
    playlist_title = None
    safe_playlist_title = None

    for _entry in videos:
        if isinstance(_entry, dict):
            if _entry.get("index") == -1:
                playlist_title = f"{_entry.get("channel_name")}: {_entry.get("title")}"
                safe_playlist_title = unidecode(
                    sanitize(_entry.get("title") or _entry.get("channel_name") or "")
                )
                # channel_id = videos[0].get("channel_id")
                break

    if playlist_title is None:
        playlist_title = videos[0][2].get("channel")
    if safe_playlist_title is None:
        safe_playlist_title = videos[0][2].get("channel")

    for ep_idx, (_ep_date, _ytdlp_ep_path, _entry) in enumerate(
        (_vid for _vid in videos if not isinstance(_vid, dict)), start=1
    ):
        if _entry is None:
            continue

        episode_num = _entry.get("index") or ep_idx
        u_id_type = _entry.get("webpage_url_domain")
        u_id = _entry.get("id")
        orig_title = _entry.get("title")

        title = get_new_title(u_id) or orig_title
        safe_title = unidecode(sanitize(title))

        episode_filename = f"{safe_playlist_title} - S{season_num:02d}E{episode_num:02d} - {safe_title}"

        _target_file = Path(library_season_dir / f"{episode_filename}.mkv")

        _log.msg(
            f"Processing episode {_log._GREEN}{episode_num}{_log._RESET} of {_log._BLUE}{total_videos}{_log._RESET}: {_log._YELLOW}{orig_title}{_log._RESET} "
        )

        _ytdlp_file = _ytdlp_ep_path / Path("video.mkv")

        if not _ytdlp_file.exists():
            _ytdlp_file = backfill_file(
                _entry, source_dir, _target_file, _ytdlp_ep_path
            )

        if not _ytdlp_file.exists():
            _ytdlp_file = download_episode(ytdlp_dir, _entry)

        if _target_file.exists():
            if _target_file.with_suffix(".nfo").exists():
                _log.msg("Found target file with nfo in library, skip processing.")
                continue

        if isinstance(_ytdlp_file, int):
            match _ytdlp_file:
                case 0:
                    _log.msg("Non-critical error, skipping")
                    continue
                case -1:
                    _log.msg("Critical error.")
                    raise RuntimeError("Critical Error.")
                case 403:
                    _log.msg("Forbidden or anti-bot.")
                    raise RuntimeError("Critical Error.")
                case _:
                    _log.msg(f"Unexpected error: {_ytdlp_file}")
                    raise RuntimeError(f"Unexpected error: {_ytdlp_file}")

        elif isinstance(_ytdlp_file, Path) and not _ytdlp_file.exists():
            _log.msg("ytdlp file missing")
            continue

        description = _entry.get("description", "")
        airdate = _ep_date.date().isoformat()

        # Episode NFO
        write_ep_nfo(
            library_season_dir,
            episode_filename,
            title,
            orig_title,
            season_num,
            episode_num,
            airdate,
            description,
            u_id_type,
            u_id,
        )

        # Episode thumbnail
        thumbnail_url = _entry.get("thumbnail", "")
        if thumbnail_url:
            _ret = download_file(
                thumbnail_url, library_season_dir, f"{episode_filename}-thumb"
            )
            if _ret:
                _log.msg(f"Downloaded thumbnail to library:\n\tDEST: {_ret}")

        for _img in itertools.chain(
            _ytdlp_ep_path.iterdir(), _ytdlp_file.parent.iterdir()
        ):
            if _img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                _dest = (library_season_dir / episode_filename).with_suffix(_img.suffix)
                if not _dest.exists():
                    try:
                        os.link(_img, _dest)
                        _log.msg(
                            f"Hardlinked image to library;\n\tSRC: {_img}\n\tDEST: {_dest}"
                        )
                    except OSError:
                        shutil.copy2(_img, _dest)
                        _log.msg(
                            f"Copied image to library;\n\tSRC: {_img}\n\tDEST: {_dest}"
                        )
                else:
                    _log.msg(f"Image exists {_dest}")

        if not _target_file.exists():
            try:
                os.link(
                    _ytdlp_file,
                    _target_file,
                )
                _log.msg(
                    f"Hardlinked video to library;\n\tSRC: {_ytdlp_file}\n\tDEST: {_target_file}"
                )
            except OSError:
                shutil.copy2(
                    _ytdlp_file,
                    _target_file,
                )
                _log.msg(
                    f"Copied video to library;\n\tSRC: {_ytdlp_file}\n\tDEST: {_target_file}"
                )
        else:
            _log.msg(f"Target video file exists {_target_file}")

        add_to_archive(_entry.get("extractor") or "", u_id, ytdlp_dir)


def create_year_episode_nfos(
    channel_dir: Path, library_show_dir: Path, source_dir: Path | None = None
):
    videos_by_year, _ = sort_videos_by_year(channel_dir)
    total_seasons = len(videos_by_year.keys())
    season_idx = 0

    for year, videos in videos_by_year.items():
        season_idx += 1
        season_num = year
        library_season_dir = library_show_dir / f"Season {season_num}"
        library_season_dir.mkdir(parents=True, exist_ok=True)

        _log.msg(
            f"Processing {_log._BLUE}{videos[0][2].get("channel")}{_log._RESET} {_log._YELLOW}Season {season_num}{_log._RESET} - {_log._GREEN}{season_idx}{_log._RESET} of {_log._BLUE}{total_seasons}{_log._RESET}"
        )

        videos.sort(key=lambda x: x[0])

        process_season_videos(
            videos, season_num, library_season_dir, source_dir, channel_dir.parent
        )


def create_playlist_episode_nfos(
    ytdlp_dir: Path,
    library_show_dir: Path,
    source_dir: Path | None,
    ytdlp_playlist_dir: Path,
):
    playlist_data = json.loads(
        (ytdlp_playlist_dir / Path("playlist.json")).read_text(encoding="utf-8")
    )

    videos: list = [playlist_data[0]]
    for _entry in playlist_data:
        if _entry.get("index") != -1:
            _entry_path = (
                ytdlp_dir
                / Path(playlist_data[0]["channel_id"])
                / Path(_entry.get("id"))
            )
            _entry_json_filename = _entry_path / Path("video.info.json")
            if _entry_json_filename.exists():
                _entry_data = json.loads(
                    (_entry_json_filename).read_text(encoding="utf-8")
                )
                _entry_data["index"] = _entry["index"]
                videos.append(
                    (
                        dateutil.parser.parse(_entry_data.get("upload_date")),
                        _entry_path,
                        _entry_data,
                    )
                )
            else:
                videos.append((None, None, None))

    process_season_videos(videos, 1, library_show_dir, source_dir, ytdlp_dir)

    return


# ----------------------------
# Main
# ----------------------------


def main():
    # Verbosity 0
    import argparse

    from fauxjson import load_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--url-file", required=True)
    parser.add_argument("--source-dir", required=False, default=None)
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--channel-library-dir", required=True)
    parser.add_argument("--playlist-library-dir", required=True)

    args = parser.parse_args()

    urls: list[str] = load_json(args.url_file) or []

    source_dir = Path(args.source_dir) if args.source_dir else None

    ytdlp_root = Path(args.download_dir)
    playlist_library_dir = Path(args.playlist_library_dir)
    channel_library_dir = Path(args.channel_library_dir)

    if urls is None:
        print("Error: Unable to populate sources.")
        return

    for _source_url in urls:
        _log.msg(f"Processing next URL: {_source_url}")
        playlist_targets: list[tuple[Path, Path | None, list[dict]]] = []

        if _source_url.lower().startswith("playlists:"):
            _source_url = _source_url.split(":", 1)[1]

            playlist_dir, playlist_srcdir, playlist_data = download_playlist(
                _source_url, ytdlp_root, source_dir
            )
            for _playlist in playlist_data:
                if not _playlist.get("index") == -1:
                    playlist_targets.append(
                        download_playlist(
                            _playlist.get("url") or "", ytdlp_root, source_dir
                        )
                    )
        else:
            playlist_targets.append(
                download_playlist(_source_url, ytdlp_root, source_dir),
            )

        for playlist_dir, playlist_srcdir, playlist_data in playlist_targets:
            if playlist_data[0]["channel_id"] == playlist_data[0]["id"]:
                # --- FULL CHANNEL

                channel_dir = playlist_dir.parent
                _log.msg(f"Channel stored at {channel_dir}")
                library_show_dir = create_tvshow_nfo(
                    channel_dir / channel_dir.name, channel_library_dir
                )
                _log.msg(
                    f"Show data for {playlist_data[0]["channel_id"]} stored to library at {library_show_dir}"
                )
                create_year_episode_nfos(channel_dir, library_show_dir, playlist_srcdir)
                _log.msg(f"Episodes and data stored to library in {library_show_dir}")

                _log.msg(f"Done with {channel_dir}.")
            else:
                # --- PLAYLIST

                playlist_name = playlist_data[0].get("title") or ""

                _log.msg(f"Playlist '{playlist_name}' stored at {playlist_dir}")

                library_show_dir = create_tvshow_nfo(playlist_dir, playlist_library_dir)
                _log.msg(
                    f"Show data for {playlist_name} stored to library at {library_show_dir}"
                )

                create_playlist_episode_nfos(
                    ytdlp_root, library_show_dir, playlist_srcdir, playlist_dir
                )
                _log.msg(f"Episodes and data stored to library in {library_show_dir}")

                _log.msg(f"Done with {playlist_dir}.")


if __name__ == "__main__":
    main()
