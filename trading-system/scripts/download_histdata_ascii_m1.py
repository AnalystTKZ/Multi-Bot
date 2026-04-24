#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener


BASE = Path(__file__).resolve().parent.parent
FOREX_DIR = BASE / "training_data" / "forex"
PAGE_URL = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{year}"
DOWNLOAD_URL = "https://www.histdata.com/get.php"

FORM_RE = re.compile(r'<form id="file_down".*?</form>', re.IGNORECASE | re.DOTALL)
INPUT_RE = re.compile(
    r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
    re.IGNORECASE,
)


def fetch_bytes(opener, url: str, data: bytes | None = None, referer: str | None = None) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
    }
    if referer:
        headers["Referer"] = referer
    request = Request(url, data=data, headers=headers)
    with opener.open(request, timeout=120) as response:
        return response.read()


def parse_download_form(html: str) -> dict[str, str]:
    match = FORM_RE.search(html)
    if not match:
        raise RuntimeError("Could not find HistData download form")
    fields = dict(INPUT_RE.findall(match.group(0)))
    required = {"tk", "date", "datemonth", "platform", "timeframe", "fxpair"}
    missing = required - fields.keys()
    if missing:
        raise RuntimeError(f"Missing form fields: {sorted(missing)}")
    return fields


def iter_ascii_rows(zip_bytes: bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = sorted(name for name in zf.namelist() if name.lower().endswith(".csv"))
        if not members:
            raise RuntimeError("Downloaded zip does not contain CSV data")
        for member in members:
            with zf.open(member) as handle:
                for raw in io.TextIOWrapper(handle, encoding="utf-8", errors="replace"):
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split(";")
                    if len(parts) < 6:
                        continue
                    dt_raw, open_, high, low, close, volume = parts[:6]
                    if len(dt_raw) != 15 or " " not in dt_raw:
                        continue
                    dt_fmt = (
                        f"{dt_raw[0:4]}-{dt_raw[4:6]}-{dt_raw[6:8]} "
                        f"{dt_raw[9:11]}:{dt_raw[11:13]}:{dt_raw[13:15]}"
                    )
                    yield [dt_fmt, open_, high, low, close, volume]


def download_year(opener, pair: str, year: int) -> bytes:
    page_url = PAGE_URL.format(pair=pair.lower(), year=year)
    html = fetch_bytes(opener, page_url).decode("utf-8", errors="replace")
    fields = parse_download_form(html)
    payload = urlencode(fields).encode("utf-8")
    return fetch_bytes(opener, DOWNLOAD_URL, data=payload, referer=page_url)


def download_pair(pair: str, years: list[int], overwrite: bool) -> Path:
    opener = build_opener(HTTPCookieProcessor())
    FOREX_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FOREX_DIR / f"{pair.upper()}_m1_histdata.csv"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists; use --overwrite to replace it")

    tmp_path = out_path.with_suffix(".csv.tmp")
    rows_written = 0
    with tmp_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Datetime", "Open", "High", "Low", "Close", "Volume"])
        for year in years:
            print(f"[{pair}] downloading {year}", flush=True)
            zip_bytes = download_year(opener, pair, year)
            count_before = rows_written
            for row in iter_ascii_rows(zip_bytes):
                writer.writerow(row)
                rows_written += 1
            print(f"[{pair}] wrote {rows_written - count_before:,} rows for {year}", flush=True)

    tmp_path.replace(out_path)
    print(f"[{pair}] complete -> {out_path} ({rows_written:,} rows)", flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HistData Generic ASCII M1 yearly archives and merge them into project CSVs."
    )
    parser.add_argument("--pairs", nargs="+", required=True, help="Forex pairs, e.g. USDZAR EURAUD")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    years = list(range(args.start_year, args.end_year + 1))
    try:
        for pair in args.pairs:
            download_pair(pair.upper(), years, overwrite=args.overwrite)
    except Exception as exc:  # pragma: no cover - CLI failure path
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
