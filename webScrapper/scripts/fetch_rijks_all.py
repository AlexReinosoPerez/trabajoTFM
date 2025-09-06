#!/usr/bin/env python3
"""
Fetch all Vincent van Gogh works from the Rijksmuseum (Linked Art Search API),
resolve IIIF manifests to get large image URLs, and download images.
Includes resume capability via a checkpoint file.

Output:
  data/rijks_all/metadata/rijks_vangogh.jsonl
  data/rijks_all/metadata/rijks_vangogh.csv
  data/rijks_all/images/rijks/<object_id>.jpg

Usage:
  python -m scripts.fetch_rijks_all --download
  python -m scripts.fetch_rijks_all --no-download
  python -m scripts.fetch_rijks_all --min-width 1500 --min-height 1500 --download
  python -m scripts.fetch_rijks_all --resume
"""

import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "rijks-vangogh-scraper/1.0 (+research/educational use)"
TIMEOUT = 30

SEARCH_URL = "https://data.rijksmuseum.nl/search/collection"
LICENSE_SHORT = "Open data (image license varies)"
LICENSE_URL = "https://data.rijksmuseum.nl/"

# Paths
OUT_BASE = os.path.join("data", "rijks_all")
META_DIR = os.path.join(OUT_BASE, "metadata")
IMG_DIR = os.path.join(OUT_BASE, "images", "rijks")
CHECKPOINT_PATH = os.path.join(OUT_BASE, "checkpoint.json")
JSONL_PATH = os.path.join(META_DIR, "rijks_vangogh.jsonl")
CSV_PATH = os.path.join(META_DIR, "rijks_vangogh.csv")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


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


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    import pandas as pd
    ensure_dir(os.path.dirname(path))
    pd.DataFrame(rows).to_csv(path, index=False)


def load_checkpoint() -> Dict[str, Any]:
    if not os.path.exists(CHECKPOINT_PATH):
        return {"next_url": SEARCH_URL, "processed_ids": []}
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"next_url": SEARCH_URL, "processed_ids": []}


def save_checkpoint(next_url: str, processed_ids: List[str]) -> None:
    ensure_dir(os.path.dirname(CHECKPOINT_PATH))
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"next_url": next_url, "processed_ids": processed_ids}, f, ensure_ascii=False, indent=2)


def dereference_linked_art(sess: requests.Session, lod_id: str) -> Optional[Dict[str, Any]]:
    try:
        r = sess.get(lod_id, headers={"Accept": "application/ld+json"}, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 406:
            r = sess.get(lod_id, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def extract_core_fields(la: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    # title
    title = None
    for label in la.get("identified_by", []):
        if label.get("type") == "Name":
            title = label.get("_label") or label.get("content")
            if title:
                break

    # accession
    accession = None
    for ident in la.get("identified_by", []):
        if ident.get("type") == "Identifier":
            accession = ident.get("content")
            break

    # date
    date = None
    prod = la.get("produced_by") or {}
    ts = prod.get("timespan") or {}
    date = ts.get("_label") or ts.get("begin_of_the_begin")

    # medium
    medium = None
    tech = prod.get("technique") or []
    if isinstance(tech, dict):
        tech = [tech]
    if tech:
        medium = ", ".join([t.get("_label") for t in tech if t.get("_label")])

    # dimensions
    dimensions = None
    for d in (la.get("dimension") or []):
        if d.get("_label"):
            dimensions = d["_label"]
            break

    return title, accession, date, medium, dimensions


def iiif_from_manifest(sess: requests.Session, manifest_url: str, min_w: int, min_h: int) -> Optional[str]:
    """
    Parse IIIF Presentation v2/v3 manifest and build a large image URL.
    Prefer IIIF Image API service when available, fallback to body @id.
    """
    try:
        m = sess.get(manifest_url, timeout=TIMEOUT)
        m.raise_for_status()
        mj = m.json()
    except Exception:
        return None

    # v2 path
    canvases = None
    sequences = (mj.get("sequences") or [])
    if sequences:
        canvases = sequences[0].get("canvases")

    # v3 path
    if not canvases:
        items = mj.get("items") or []
        if items and isinstance(items[0], dict):
            canvases = items

    if not canvases:
        return None

    # choose the first canvas that meets size constraints (if present)
    best_url = None
    for c in canvases:
        width = c.get("width") or 0
        height = c.get("height") or 0

        # v2 image resource
        svc_id = None
        direct_url = None
        images = c.get("images") or []
        if images:
            res = images[0].get("resource") or {}
            svc = res.get("service") or {}
            svc_id = svc.get("@id") or svc.get("id")
            direct_url = res.get("@id") or res.get("id")

        # v3 image body
        if not direct_url:
            citems = c.get("items") or []
            if citems and citems[0].get("items"):
                body = citems[0]["items"][0].get("body") or {}
                if isinstance(body, list):
                    body = body[0]
                svc = body.get("service") or []
                if isinstance(svc, dict):
                    svc = [svc]
                if svc:
                    svc_id = svc[0].get("id") or svc[0].get("@id")
                direct_url = body.get("id") or body.get("@id")
                width = body.get("width") or width
                height = body.get("height") or height

        # build final URL
        if svc_id:
            svc_id = svc_id.rstrip("/")
            candidate = f"{svc_id}/full/2000,/0/default.jpg"
        else:
            candidate = direct_url

        if not candidate:
            continue

        # check size threshold if available
        if (min_w and width and width < min_w) or (min_h and height and height < min_h):
            if svc_id:
                best_url = candidate
                break
            continue

        best_url = candidate
        break

    return best_url


def normalize_record(
    object_id: str,
    title: Optional[str],
    date: Optional[str],
    medium: Optional[str],
    dimensions: Optional[str],
    accession_number: Optional[str],
    object_url: Optional[str],
    iiif_image_url: Optional[str],
    iiif_manifest_url: Optional[str],
) -> Dict[str, Any]:
    return {
        "label": 1,  # positive (Van Gogh)
        "source": "rijks",
        "object_id": object_id,
        "title": title,
        "date": date,
        "medium": medium,
        "dimensions": dimensions,
        "artist": "Vincent van Gogh",
        "museum": "Rijksmuseum",
        "accession_number": accession_number,
        "object_url": object_url,
        "iiif_image_url": iiif_image_url,
        "iiif_manifest_url": iiif_manifest_url,
        "license": LICENSE_SHORT,
        "license_url": LICENSE_URL,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="Download images")
    p.add_argument("--no-download", action="store_true", help="Skip downloads")
    p.add_argument("--min-width", type=int, default=0, help="Preferred minimum image width")
    p.add_argument("--min-height", type=int, default=0, help="Preferred minimum image height")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_download:
        args.download = False

    ensure_dir(META_DIR)
    ensure_dir(IMG_DIR)

    sess = build_session()

    # checkpoint
    ckpt = load_checkpoint() if args.resume else {"next_url": SEARCH_URL, "processed_ids": []}
    next_url = ckpt.get("next_url", SEARCH_URL)
    processed_ids = set(ckpt.get("processed_ids", []))

    # fetch loop: iterate every page until there is no "next"
    all_rows: List[Dict[str, Any]] = []
    params = {"creator": "Vincent van Gogh", "imageAvailable": "true"}

    while True:
        try:
            r = sess.get(next_url, params=params if next_url == SEARCH_URL else None, timeout=TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[ERROR] Search request failed: {e}")
            break

        data = r.json()
        items = data.get("orderedItems") or []
        if not items:
            print("[INFO] No more items in this page.")
        for it in items:
            lod_id = it.get("id")
            if not lod_id:
                continue
            oid = lod_id.split("/")[-1]
            if oid in processed_ids:
                continue

            la = dereference_linked_art(sess, lod_id)
            if not la:
                continue

            title, accession, date, medium, dimensions = extract_core_fields(la)
            manifest_url = None

            # discover manifest in Linked Art
            rep = la.get("representation") or []
            for rsrc in rep:
                srv = rsrc.get("service")
                if isinstance(srv, dict) and "id" in srv and "iiif" in (srv.get("conforms_to") or ""):
                    manifest_url = srv.get("id")
                    break
                if rsrc.get("id", "").endswith("manifest"):
                    manifest_url = rsrc["id"]
                    break
            if not manifest_url:
                for s in (la.get("subject_of") or []):
                    if s.get("id", "").endswith("manifest"):
                        manifest_url = s["id"]
                        break

            image_url = None
            if manifest_url:
                image_url = iiif_from_manifest(sess, manifest_url, args.min_width, args.min_height)

            row = normalize_record(
                object_id=oid,
                title=title,
                date=date,
                medium=medium,
                dimensions=dimensions,
                accession_number=accession,
                object_url=la.get("id"),
                iiif_image_url=image_url,
                iiif_manifest_url=manifest_url,
            )
            all_rows.append(row)
            processed_ids.add(oid)

            # save checkpoint periodically
            if len(all_rows) % 25 == 0:
                save_checkpoint(next_url, sorted(processed_ids))
                print(f"[INFO] Checkpoint saved ({len(processed_ids)} processed).")
                time.sleep(0.3)

        # pagination
        nxt = (data.get("next") or {}).get("id")
        if not nxt:
            print("[INFO] Reached the end of pagination.")
            break
        next_url = nxt
        params = None  # follow absolute next link
        save_checkpoint(next_url, sorted(processed_ids))
        time.sleep(0.3)

    if not all_rows:
        print("[WARN] No records collected.")
        return

    # Save metadata
    save_jsonl(all_rows, JSONL_PATH)
    save_csv(all_rows, CSV_PATH)
    print(f"[INFO] Saved metadata:\n  {JSONL_PATH}\n  {CSV_PATH}")

    # Optional image download
    if args.download:
        downloaded = 0
        for row in all_rows:
            url = row.get("iiif_image_url")
            if not url:
                continue
            oid = row["object_id"]
            ext = ".jpg" if url.lower().endswith(".jpg") or ".jpeg" in url.lower() else (".tif" if ".tif" in url.lower() else ".bin")
            out_path = os.path.join(IMG_DIR, f"{oid}{ext}")
            if os.path.exists(out_path):
                continue
            try:
                with sess.get(url, stream=True, timeout=TIMEOUT) as resp:
                    resp.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                downloaded += 1
                if downloaded % 25 == 0:
                    print(f"[INFO] Downloaded images: {downloaded}")
            except requests.RequestException:
                continue
        print(f"[INFO] Total downloaded images: {downloaded}")


def fetch(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Fetch Van Gogh records from Rijksmuseum Linked Art API up to `limit`.
    Returns a list of normalized dicts (no downloads, no checkpoint).
    Useful for being imported by other scripts, e.g., scripts.fetch_vangogh_dataset.
    """
    sess = build_session()
    results: List[Dict[str, Any]] = []
    next_url = SEARCH_URL
    params = {"creator": "Vincent van Gogh", "imageAvailable": "true"}

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
            if len(results) >= limit:
                break
            lod_id = it.get("id")
            if not lod_id:
                continue
            la = dereference_linked_art(sess, lod_id)
            if not la:
                continue

            title, accession, date, medium, dimensions = extract_core_fields(la)

            # discover manifest and try to map an image URL
            manifest_url = None
            rep = la.get("representation") or []
            for rsrc in rep:
                srv = rsrc.get("service")
                if isinstance(srv, dict) and "id" in srv and "iiif" in (srv.get("conforms_to") or ""):
                    manifest_url = srv.get("id")
                    break
                if rsrc.get("id", "").endswith("manifest"):
                    manifest_url = rsrc["id"]
                    break
            if not manifest_url:
                for s in (la.get("subject_of") or []):
                    if s.get("id", "").endswith("manifest"):
                        manifest_url = s["id"]
                        break

            image_url = iiif_from_manifest(sess, manifest_url, 0, 0) if manifest_url else None

            oid = lod_id.split("/")[-1]
            results.append(
                normalize_record(
                    object_id=oid,
                    title=title,
                    date=date,
                    medium=medium,
                    dimensions=dimensions,
                    accession_number=accession,
                    object_url=la.get("id"),
                    iiif_image_url=image_url,
                    iiif_manifest_url=manifest_url,
                )
            )

        nxt = (data.get("next") or {}).get("id")
        if not nxt:
            break
        next_url = nxt
        params = None
        time.sleep(0.2)

    return results


if __name__ == "__main__":
    main()
