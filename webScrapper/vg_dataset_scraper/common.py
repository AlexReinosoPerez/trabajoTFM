import os
import time
import json
import pathlib
from typing import Dict, Any, Iterable, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "vg-dataset-scraper/1.0 (+for research/educational use)"
TIMEOUT = 30

def build_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    sess.headers.update({"User-Agent": USER_AGENT})
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def save_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import pandas as pd
    ensure_dir(os.path.dirname(path))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

def download_image(sess: requests.Session, url: str, out_path: str) -> bool:
    ensure_dir(os.path.dirname(out_path))
    try:
        with sess.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except requests.RequestException:
        return False

def normalize_record(
    source: str,
    object_id: str,
    title: Optional[str],
    date: Optional[str],
    medium: Optional[str],
    dimensions: Optional[str],
    artist: Optional[str],
    museum: Optional[str],
    accession_number: Optional[str],
    object_url: Optional[str],
    iiif_image_url: Optional[str],
    iiif_manifest_url: Optional[str],
    license_short: str,
    license_url: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec = {
        "source": source,
        "object_id": str(object_id),
        "title": title,
        "date": date,
        "medium": medium,
        "dimensions": dimensions,
        "artist": artist,
        "museum": museum,
        "accession_number": accession_number,
        "object_url": object_url,
        "iiif_image_url": iiif_image_url,
        "iiif_manifest_url": iiif_manifest_url,
        "license": license_short,     # e.g., CC0, Public Domain, Non-Commercial
        "license_url": license_url,
    }
    if extra:
        rec["extra"] = extra
    return rec
