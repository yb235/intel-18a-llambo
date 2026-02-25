#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data/raw/source_manifest.csv"
DOWNLOAD_DIR = REPO_ROOT / "data/raw/downloads"
HASHES_OUT = REPO_ROOT / "data/interim/source_hashes.csv"


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def infer_extension(content_type: str, url: str) -> str:
    ctype = (content_type or "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return ".pdf"
    if "json" in ctype:
        return ".json"
    return ".html"


def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    HASHES_OUT.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with MANIFEST.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    out_rows: list[dict[str, str]] = []
    for row in rows:
        source_id = row["source_id"]
        url = row["url"].strip()

        if url.startswith("mcporter:"):
            out_rows.append(
                {
                    "source_id": source_id,
                    "url": url,
                    "status": "skipped_non_http",
                    "http_status": "",
                    "content_type": row.get("content_type", ""),
                    "bytes": "",
                    "sha256": "",
                    "local_path": "",
                    "error": "",
                }
            )
            continue

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "intel-18a-llambo-ingest/0.1 (+https://github.com/)",
                "Accept": "text/html,application/pdf,application/json,*/*",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read()
                http_status = str(getattr(response, "status", ""))
                content_type = response.headers.get("Content-Type", "")
                ext = infer_extension(content_type, url)
                out_path = DOWNLOAD_DIR / f"{source_id}{ext}"
                out_path.write_bytes(payload)
                out_rows.append(
                    {
                        "source_id": source_id,
                        "url": url,
                        "status": "ok",
                        "http_status": http_status,
                        "content_type": content_type,
                        "bytes": str(len(payload)),
                        "sha256": sha256_bytes(payload),
                        "local_path": str(out_path.relative_to(REPO_ROOT)),
                        "error": "",
                    }
                )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            out_rows.append(
                {
                    "source_id": source_id,
                    "url": url,
                    "status": "error",
                    "http_status": "",
                    "content_type": row.get("content_type", ""),
                    "bytes": "",
                    "sha256": "",
                    "local_path": "",
                    "error": str(exc),
                }
            )

    fieldnames = [
        "source_id",
        "url",
        "status",
        "http_status",
        "content_type",
        "bytes",
        "sha256",
        "local_path",
        "error",
    ]
    with HASHES_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    ok = sum(1 for row in out_rows if row["status"] == "ok")
    err = sum(1 for row in out_rows if row["status"] == "error")
    print(f"Wrote {HASHES_OUT} (ok={ok}, error={err}).")


if __name__ == "__main__":
    main()
