"""
Cleveland Museum of Art Open Access API. Docs: https://openaccess-api.clevelandart.org/
Use GET /api/artworks with artists filter, has_image=1, cc0 filter, and fields.
Images include 'web', 'print', 'full' under images.* (CC0).
"""
from typing import List, Dict, Any
import requests
from ..common import build_session, normalize_record

SEARCH_URL = "https://openaccess-api.clevelandart.org/api/artworks/"
LICENSE_SHORT = "CC0"
LICENSE_URL = "https://www.clevelandart.org/open-access"

FIELDS = ",".join([
    "id","accession_number","title","creators","creation_date",
    "technique","measurements","department","url","images"
])

def fetch(limit: int = 1000) -> List[Dict[str, Any]]:
    sess = build_session()
    results: List[Dict[str, Any]] = []
    skip = 0
    page_size = 1000  # CMA allows up to 1000 per page

    while True:
        params = {
            "artists": "Vincent van Gogh",
            "has_image": 1,
            "cc0": "",               # flag presence means filter CC0
            "limit": min(page_size, limit - len(results)),
            "skip": skip,
            "fields": FIELDS,
        }
        r = sess.get(SEARCH_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("data", [])
        if not items:
            break
        for it in items:
            imgs = (it.get("images") or {}).get("print") or (it.get("images") or {}).get("web")
            # prefer 'full' tif if present
            full = (it.get("images") or {}).get("full") or {}
            url_image = (full.get("url") or (it.get("images") or {}).get("print", {}).get("url")
                         or (it.get("images") or {}).get("web", {}).get("url"))
            artist = None
            creators = it.get("creators") or []
            if creators:
                # usually first has role "artist"
                artist = creators[0].get("description") or creators[0].get("name")

            rec = normalize_record(
                source="cma",
                object_id=it.get("id"),
                title=it.get("title"),
                date=it.get("creation_date"),
                medium=it.get("technique"),
                dimensions=it.get("measurements"),
                artist=artist or "Vincent van Gogh",
                museum="Cleveland Museum of Art",
                accession_number=it.get("accession_number"),
                object_url=it.get("url"),
                iiif_image_url=url_image,
                iiif_manifest_url=None,  # CMA uses static CDN URLs; no IIIF manifest by default
                license_short=LICENSE_SHORT,
                license_url=LICENSE_URL,
                extra={"department": it.get("department")},
            )
            results.append(rec)
            if len(results) >= limit:
                return results
        skip += len(items)
    return results
