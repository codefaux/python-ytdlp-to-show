from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigData:
    urls: list[str]
    source_dir: Path | None
    ytdlp_root: Path
    channel_library_dir: Path
    playlist_library_dir: Path
    preserve_source: bool

    def __init__(self):
        import argparse
        from fauxjson import load_json

        parser = argparse.ArgumentParser()
        parser.add_argument("--url-file", required=True)
        parser.add_argument("--source-dir", required=False, default=None)
        parser.add_argument("--download-dir", required=True)
        parser.add_argument("--channel-library-dir", required=True)
        parser.add_argument("--playlist-library-dir", required=True)
        parser.add_argument("--preserve-source", required=False, default=False)

        args = parser.parse_args()

        self.urls = load_json(args.url_file) or []
        self.source_dir = Path(args.source_dir) if args.source_dir else None
        self.ytdlp_root = Path(args.download_dir)
        self.playlist_library_dir = Path(args.playlist_library_dir)
        self.channel_library_dir = Path(args.channel_library_dir)


config = ConfigData()
