#!/usr/bin/env python3
"""
Collect a negative set (non-Van Gogh) from open-access museum APIs:
- The Met (CC0)
- Art Institute of Chicago (AIC) (public domain)
- Cleveland Museum of Art (CMA) (CC0)
- Rijksmuseum (Linked Art Search; image license varies)

Target: ~N images balanced across artists and sources.
It saves unified metadata (CSV + JSONL) and can optionally download images.

Usage examples:
  python -m scripts.fetch_negatives --target 1500 --download
  python -m scripts.fetch_negatives --artists "Paul Gauguin" "Paul Signac" --target 600
  python -m scripts.fetch_negatives --sources met aic --target 800 --per-artist-cap 300 --download
"""

import os
import time
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------
# Config
# --------------------------

DEFAULT_ARTISTS = [
    # Post-Impressionists and close comparators
    "Paul Gauguin",
    "Paul Signac",
    "Camille Pissarro",
    "Georges Seurat",
    "Henri de Toulouse-Lautrec",
    "Émile Bernard",
    "Claude Monet",
    "Paul Cézanne",
]

DEFAULT_SOURCES = ["met", "aic", "cma", "rijks"]

USER_AGENT = "vg-negatives-scraper/1.0 (+research/educational use)"
TIMEOUT = 30
AIC_PER_PAGE_SLEEP = 0.7
AIC_MAX_403_RETRIES = 6
AIC_INITIAL_BACKOFF = 5.0

# --------------------------
# Helpers
# --------------------------

def build_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    sess.headers.update({"User-Agent": USER_AGENT})
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            import json
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import pandas as pd
    ensure_dir(os.path.dirname(path))
    pd.DataFrame(rows).to_csv(path, index=False)

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
        "label": 0,  # negative
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
        "license": license_short,
        "license_url": license_url,
    }
    if extra:
        rec["extra"] = extra
    return rec

# --------------------------
# Source: The Met
# --------------------------

def met_fetch_by_artist(sess: requests.Session, artist: str, limit: int) -> List[Dict[str, Any]]:
    SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
    OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"
    LICENSE_SHORT = "CC0"
    LICENSE_URL = "https://www.metmuseum.org/about-the-met/policies-and-documents/open-access"

    params = {"q": artist, "hasImages": "true", "artistOrCulture": "true"}
    try:
        r = sess.get(SEARCH_URL, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        ids = r.json().get("objectIDs") or []
    except requests.RequestException:
        ids = []
    if not ids:
        return []

    ids = ids[:limit]
    out: List[Dict[str, Any]] = []

    def _fetch(oid: int) -> Optional[Dict[str, Any]]:
        try:
            rr = sess.get(OBJECT_URL.format(oid), timeout=TIMEOUT)
            rr.raise_for_status()
            return rr.json()
        except requests.RequestException:
            return None

    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_fetch, oid): oid for oid in ids}
        for fut in as_completed(futs):
            obj = fut.result()
            if not obj:
                continue
            if not obj.get("isPublicDomain") or not obj.get("primaryImage"):
                continue
            out.append(
                normalize_record(
                    source="met",
                    object_id=obj.get("objectID"),
                    title=obj.get("title"),
                    date=obj.get("objectDate"),
                    medium=obj.get("medium"),
                    dimensions=obj.get("dimensions"),
                    artist=obj.get("artistDisplayName"),
                    museum="The Metropolitan Museum of Art",
                    accession_number=obj.get("accessionNumber"),
                    object_url=obj.get("objectURL"),
                    iiif_image_url=obj.get("primaryImage"),
                    iiif_manifest_url=None,
                    license_short=LICENSE_SHORT,
                    license_url=LICENSE_URL,
                    extra={"department": obj.get("department")},
                )
            )
            if len(out) >= limit:
                break
    return out[:limit]

# --------------------------
# Source: AIC
# --------------------------

def aic_fetch_by_artist(sess: requests.Session, artist: str, limit: int) -> List[Dict[str, Any]]:
    SEARCH_URL = "https://api.artic.edu/api/v1/artworks/search"
    FIELDS = ",".join([
        "id","title","artist_title","date_display","medium_display","dimensions",
        "credit_line","image_id","iiif_url","is_public_domain","api_link","department_title"
    ])
    LICENSE_SHORT = "Public domain (AIC); data under AIC terms"
    LICENSE_URL = "https://www.artic.edu/open-access/public-api"

    results: List[Dict[str, Any]] = []
    page = 1
    backoff = AIC_INITIAL_BACKOFF

    while len(results) < limit:
        params = {
            "q": f'artist_title:"{artist}"',
            "fields": FIELDS,
            "limit": min(100, limit - len(results)),
            "page": page,
        }
        for attempt in range(AIC_MAX_403_RETRIES + 1):
            try:
                r = sess.get(SEARCH_URL, params=params, timeout=TIMEOUT)
                if r.status_code == 403:
                    if attempt >= AIC_MAX_403_RETRIES:
                        r.raise_for_status()
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                r.raise_for_status()
                break
            except requests.HTTPError:
                raise
            except requests.RequestException:
                time.sleep(3)
                continue

        data = r.json()
        arr = data.get("data", [])
        if not arr:
            break

        for it in arr:
            if not it.get("is_public_domain") or not it.get("image_id") or not it.get("iiif_url"):
                continue
            iiif = f"{it['iiif_url'].rstrip('/')}/{it['image_id']}/full/2000,/0/default.jpg"
            results.append(
                normalize_record(
                    source="aic",
                    object_id=it["id"],
                    title=it.get("title"),
                    date=it.get("date_display"),
                    medium=it.get("medium_display"),
                    dimensions=it.get("dimensions"),
                    artist=it.get("artist_title"),
                    museum="Art Institute of Chicago",
                    accession_number=None,
                    object_url=it.get("api_link"),
                    iiif_image_url=iiif,
                    iiif_manifest_url=f"{it['iiif_url'].rstrip('/')}/{it['image_id']}/manifest.json",
                    license_short=LICENSE_SHORT,
                    license_url=LICENSE_URL,
                    extra={"department": it.get("department_title")},
                )
            )
            if len(results) >= limit:
                break

        time.sleep(AIC_PER_PAGE_SLEEP)
        info = data.get("pagination") or data.get("info", {}).get("pagination")
        if info and page >= info.get("total_pages", page):
            break
        page += 1

    return results[:limit]

# --------------------------
# Source: CMA
# --------------------------

def cma_fetch_by_artist(sess: requests.Session, artist: str, limit: int) -> List[Dict[str, Any]]:
    SEARCH_URL = "https://openaccess-api.clevelandart.org/api/artworks/"
    LICENSE_SHORT = "CC0"
    LICENSE_URL = "https://www.clevelandart.org/open-access"

    results: List[Dict[str, Any]] = []
    skip = 0
    while len(results) < limit:
        params = {
            "artists": artist,
            "has_image": 1,
            "cc0": "",
            "limit": min(1000, limit - len(results)),
            "skip": skip,
            "fields": ",".join([
                "id","accession_number","title","creators","creation_date",
                "technique","measurements","department","url","images"
            ]),
        }
        try:
            r = sess.get(SEARCH_URL, params=params, timeout=TIMEOUT)
            r.raise_for_status()
        except requests.RequestException:
            break

        data = r.json()
        items = data.get("data", [])
        if not items:
            break

        for it in items:
            images = it.get("images") or {}
            url_image = (images.get("full", {}) or {}).get("url") or \
                        (images.get("print", {}) or {}).get("url") or \
                        (images.get("web", {}) or {}).get("url")
            if not url_image:
                continue
            creators = it.get("creators") or []
            artist_name = None
            if creators:
                artist_name = creators[0].get("description") or creators[0].get("name")

            results.append(
                normalize_record(
                    source="cma",
                    object_id=it.get("id"),
                    title=it.get("title"),
                    date=it.get("creation_date"),
                    medium=it.get("technique"),
                    dimensions=it.get("measurements"),
                    artist=artist_name or artist,
                    museum="Cleveland Museum of Art",
                    accession_number=it.get("accession_number"),
                    object_url=it.get("url"),
                    iiif_image_url=url_image,
                    iiif_manifest_url=None,
                    license_short=LICENSE_SHORT,
                    license_url=LICENSE_URL,
                    extra={"department": it.get("department")},
                )
            )
            if len(results) >= limit:
                break

        skip += len(items)

    return results[:limit]

# --------------------------
# Source: Rijks (Linked Art Search)
# --------------------------

def rijks_fetch_by_artist(sess: requests.Session, artist: str, limit: int) -> List[Dict[str, Any]]:
    SEARCH_URL = "https://data.rijksmuseum.nl/search/collection"
    LICENSE_SHORT = "Open data (image license varies)"
    LICENSE_URL = "https://data.rijksmuseum.nl/"

    results: List[Dict[str, Any]] = []
    next_url = SEARCH_URL
    params = {"creator": artist, "imageAvailable": "true"}

    def _dereference(lod_id: str) -> Optional[Dict[str, Any]]:
        try:
            rr = sess.get(lod_id, headers={"Accept": "application/ld+json"}, timeout=TIMEOUT, allow_redirects=True)
            if rr.status_code == 406:
                rr = sess.get(lod_id, timeout=TIMEOUT, allow_redirects=True)
            rr.raise_for_status()
            return rr.json()
        except Exception:
            return None

    def _extract_fields(la: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        title = None
        for label in la.get("identified_by", []):
            if label.get("type") == "Name":
                title = label.get("_label") or label.get("content")
                if title:
                    break
        accession = None
        for ident in la.get("identified_by", []):
            if ident.get("type") == "Identifier":
                accession = ident.get("content")
                break
        date = None
        prod = la.get("produced_by") or {}
        ts = prod.get("timespan") or {}
        date = ts.get("_label") or ts.get("begin_of_the_begin")
        medium = None
        tech = prod.get("technique") or []
        if isinstance(tech, dict):
            tech = [tech]
        if tech:
            medium = ", ".join([t.get("_label") for t in tech if t.get("_label")])
        dimensions = None
        for m in (la.get("dimension") or []):
            if m.get("_label"):
                dimensions = m["_label"]
                break
        manifest_url = None
        for r in (la.get("representation") or []):
            srv = r.get("service")
            if isinstance(srv, dict) and "id" in srv and "iiif" in (srv.get("conforms_to") or ""):
                manifest_url = srv.get("id")
                break
            if r.get("id", "").endswith("manifest"):
                manifest_url = r["id"]
                break
        if not manifest_url:
            for s in (la.get("subject_of") or []):
                if s.get("id", "").endswith("manifest"):
                    manifest_url = s["id"]
                    break
        iiif_image = None
        return title, date, medium, dimensions, accession, manifest_url

    while len(results) < limit:
        try:
            r = sess.get(next_url, params=params if next_url == SEARCH_URL else None, timeout=TIMEOUT)
            r.raise_for_status()
        except requests.RequestException:
            break
        data = r.json()
        items = data.get("orderedItems") or []
        if not items:
            break

        for it in items:
            lod_id = it.get("id")
            if not lod_id:
                continue
            la = _dereference(lod_id)
            if not la:
                continue
            title, date, medium, dimensions, accession, manifest = _extract_fields(la)
            results.append(
                normalize_record(
                    source="rijks",
                    object_id=lod_id.split("/")[-1],
                    title=title,
                    date=date,
                    medium=medium,
                    dimensions=dimensions,
                    artist=artist,
                    museum="Rijksmuseum",
                    accession_number=accession,
                    object_url=la.get("id"),
                    iiif_image_url=None,
                    iiif_manifest_url=manifest,
                    license_short=LICENSE_SHORT,
                    license_url=LICENSE_URL,
                    extra={"linked_art": True},
                )
            )
            if len(results) >= limit:
                break

        nxt = (data.get("next") or {}).get("id")
        if not nxt:
            break
        next_url = nxt
        params = None

    return results[:limit]

# --------------------------
# Orchestration
# --------------------------

def fetch_from_source(sess: requests.Session, source: str, artist: str, limit: int) -> List[Dict[str, Any]]:
    if source == "met":
        return met_fetch_by_artist(sess, artist, limit)
    if source == "aic":
        return aic_fetch_by_artist(sess, artist, limit)
    if source == "cma":
        return cma_fetch_by_artist(sess, artist, limit)
    if source == "rijks":
        return rijks_fetch_by_artist(sess, artist, limit)
    return []

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, help="Which sources to use")
    p.add_argument("--artists", nargs="+", default=DEFAULT_ARTISTS, help="Artists to fetch as negatives")
    p.add_argument("--target", type=int, default=1500, help="Target number of negatives (approx)")
    p.add_argument("--per-artist-cap", type=int, default=None, help="Hard cap per artist (optional)")
    p.add_argument("--outdir", default="data/negatives", help="Base output directory")
    p.add_argument("--download", action="store_true", help="Download images")
    p.add_argument("--max-images", type=int, default=2_000_000_000, help="Safety cap for images")
    return p.parse_args()

def main():
    args = parse_args()
    sess = build_session()
    ensure_dir(args.outdir)

    # compute per-artist budget
    n_artists = len(args.artists)
    base_quota = math.ceil(args.target / max(1, n_artists))
    if args.per-artist-cap:
        base_quota = min(base_quota, args.per-artist-cap)

    all_rows: List[Dict[str, Any]] = []
    for artist in args.artists:
        artist_quota = base_quota
        got_for_artist = 0
        print(f"\n=== Artist: {artist} (quota ~{artist_quota}) ===")
        # round-robin across sources to diversify
        while got_for_artist < artist_quota and args.sources:
            progressed = False
            for src in args.sources:
                need = artist_quota - got_for_artist
                if need <= 0:
                    break
                take = max(1, min(100, need))
                print(f"  [{src}] fetching up to {take} ...")
                try:
                    rows = fetch_from_source(sess, src, artist, limit=take)
                except Exception as e:
                    print(f"  [WARN] {src} failed for {artist}: {e}")
                    rows = []
                if rows:
                    all_rows.extend(rows)
                    got_for_artist += len(rows)
                    progressed = True
                    print(f"    -> got {len(rows)} (artist total: {got_for_artist})")
                # be gentle to APIs
                time.sleep(0.4)
            if not progressed:
                print(f"  No more results for {artist}.")
                break

    if not all_rows:
        print("[ERROR] No negatives collected.")
        return

    # save metadata
    meta_dir = os.path.join(args.outdir, "metadata")
    ensure_dir(meta_dir)
    csv_path = os.path.join(meta_dir, "negatives_master.csv")
    jsonl_path = os.path.join(meta_dir, "negatives_master.jsonl")
    save_csv(csv_path, all_rows)
    save_jsonl(jsonl_path, all_rows)
    print(f"\n[INFO] Saved metadata:\n  {csv_path}\n  {jsonl_path}")

    # optional downloads
    if args.download:
        from tqdm import tqdm
        img_base = os.path.join(args.outdir, "images")
        ensure_dir(img_base)
        downloaded = 0
        for row in tqdm(all_rows, desc="Downloading images"):
            url = row.get("iiif_image_url")
            # For Rijks entries we often only have a manifest; skip actual image fetch here
            if not url:
                continue
            src = row["source"]
            obj_id = row["object_id"]
            ext = ".jpg" if url.lower().endswith(".jpg") or ".jpeg" in url.lower() else (".tif" if ".tif" in url.lower() else ".bin")
            out = os.path.join(img_base, src, f"{obj_id}{ext}")
            if os.path.exists(out):
                continue
            if download_image(sess, url, out):
                downloaded += 1
                if downloaded >= args.max_images:
                    break
        print(f"[INFO] Downloaded images: {downloaded}")

if __name__ == "__main__":
    main()
