#!/usr/bin/env python3
"""
Quick AIC fetcher using the exact working query you tested:
- q = "Vincent van Gogh"
- query[term][is_public_domain] = true
- fields include image_id + iiif_url
Paginates gently, saves to the same schema, and can download images.
"""

import os, time, random, json
from typing import List, Dict, Any
import requests
from vg_dataset_scraper.common import ensure_dir, build_session, save_csv, save_jsonl, download_image

START_PAGE   = int(os.environ.get("AIC_START_PAGE", "1"))
MAX_PAGES    = os.environ.get("AIC_MAX_PAGES")
MAX_PAGES    = int(MAX_PAGES) if (MAX_PAGES and MAX_PAGES.isdigit()) else None
MAX_403_PAGE = int(os.environ.get("AIC_MAX_403_PER_PAGE", "6"))

SEARCH_URL = "https://api.artic.edu/api/v1/artworks/search"
FIELDS = "id,title,artist_title,date_display,medium_display,dimensions,credit_line,image_id,iiif_url,is_public_domain,api_link,department_title"

def fetch_aic(limit: int = 400, per_page: int = 50, sleep_s: float = 1.6) -> List[Dict[str, Any]]:
    sess = build_session()
    sess.headers.update({
        "Referer": "https://www.artic.edu/",
        "User-Agent": "aic-quick/1.0 (+research/educational use)",
    })
    out: List[Dict[str, Any]] = []
    page = START_PAGE
    pages_done = 0
    while len(out) < limit:
        if MAX_PAGES is not None and pages_done >= MAX_PAGES:
            break
        params = {
            "q": "Vincent van Gogh",
            "query[term][is_public_domain]": "true",
            "fields": FIELDS,
            "limit": min(per_page, limit - len(out)),
            "page": page,
        }
        tries_403 = 0
        while True:
            r = sess.get(SEARCH_URL, params=params, timeout=30)
            print(f"[DEBUG] GET {r.url} -> {r.status_code}")
            if r.status_code == 403:
                tries_403 += 1
                if tries_403 >= MAX_403_PAGE:
                    print(f"[WARN] Skipping page {page} after {tries_403}x 403.")
                    break  # pasamos a la siguiente página
                wait = 8.0 + random.uniform(0, 2.0)
                print(f"[WARN] 403 rate limit. Sleeping {wait:.1f}s and retrying page {page}...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            arr = data.get("data") or []
            print(f"[DEBUG] page={page} items={len(arr)}")
            for it in arr:
                if not it.get("is_public_domain"):
                    continue
                if not it.get("image_id") or not it.get("iiif_url"):
                    continue
                iiif = f"{it['iiif_url'].rstrip('/')}/{it['image_id']}/full/2000,/0/default.jpg"
                out.append({
                    "source": "aic",
                    "object_id": it["id"],
                    "title": it.get("title"),
                    "date": it.get("date_display"),
                    "medium": it.get("medium_display"),
                    "dimensions": it.get("dimensions"),
                    "artist": it.get("artist_title"),
                    "museum": "Art Institute of Chicago",
                    "accession_number": None,
                    "object_url": it.get("api_link"),
                    "iiif_image_url": iiif,
                    "iiif_manifest_url": f"{it['iiif_url'].rstrip('/')}/{it['image_id']}/manifest.json",
                    "license": "Public domain (AIC)",
                    "license_url": "https://www.artic.edu/open-access/public-api",
                    "extra": {"department": it.get("department_title")},
                })
                if len(out) >= limit:
                    break
            break  # salimos del while True de la página actual

        pages_done += 1
        page += 1
        time.sleep(sleep_s + random.uniform(0, 1.0))
    return out


def main():
    limit = int(os.environ.get("AIC_LIMIT", "400"))
    per_page = int(os.environ.get("AIC_PER_PAGE", "50"))
    sleep_s = float(os.environ.get("AIC_SLEEP", "1.6"))
    download = os.environ.get("AIC_DOWNLOAD", "0") == "1"

    rows = fetch_aic(limit=limit, per_page=per_page, sleep_s=sleep_s)
    if not rows:
        print("[ERROR] No rows from AIC.")
        return

    ensure_dir("data/metadata")
    ensure_dir("data/images/aic")
    # append to your master files
    master_csv = "data/metadata/van_gogh_master.csv"
    master_jsonl = "data/metadata/van_gogh_master.jsonl"

    # append-safe: leemos si existen, si no, creamos
    import pandas as pd
    df_new = pd.DataFrame(rows)
    if os.path.exists(master_csv):
        df_old = pd.read_csv(master_csv)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(master_csv, index=False)
    save_jsonl(df_all.to_dict(orient="records"), master_jsonl)
    print(f"[OK] Saved/updated: {master_csv}  &  {master_jsonl}")

    if download:
        import tqdm
        sess = build_session()
        downloaded = 0
        for row in tqdm.tqdm(rows, desc="Downloading AIC images"):
            url = row.get("iiif_image_url")
            if not url: 
                continue
            oid = row["object_id"]
            out_path = os.path.join("data", "images", "aic", f"{oid}.jpg")
            if os.path.exists(out_path):
                continue
            if download_image(sess, url, out_path):
                downloaded += 1
        print(f"[INFO] Downloaded images: {downloaded}")

if __name__ == "__main__":
    main()
