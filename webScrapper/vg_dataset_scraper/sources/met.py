"""
The Met Collection API (no key needed). Docs: https://metmuseum.github.io/
We use search with hasImages=true & artistOrCulture=true, then resolve objects.
License: CC0 for Open Access images. Check fields isPublicDomain and primaryImage.
"""
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from ..common import build_session, normalize_record

SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"
LICENSE_SHORT = "CC0"
LICENSE_URL = "https://www.metmuseum.org/about-the-met/policies-and-documents/open-access"

def _fetch_object(sess: requests.Session, oid: int) -> Optional[Dict[str, Any]]:
    try:
        r = sess.get(OBJECT_URL.format(oid), timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None

def fetch(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    sess = build_session()
    # Search Van Gogh with images
    params = {
        "q": "Vincent van Gogh",
        "hasImages": "true",
        "artistOrCulture": "true",
    }
    ids = []
    try:
        resp = sess.get(SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        ids = data.get("objectIDs") or []
    except requests.RequestException:
        ids = []

    if limit:
        ids = ids[:limit]

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_fetch_object, sess, oid): oid for oid in ids}
        for fut in as_completed(futs):
            obj = fut.result()
            if not obj:
                continue
            # Only public domain with an image
            if not obj.get("isPublicDomain") or not obj.get("primaryImage"):
                continue
            rec = normalize_record(
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
                iiif_manifest_url=None,  # Met does not expose IIIF manifests via this API
                license_short=LICENSE_SHORT,
                license_url=LICENSE_URL,
                extra={
                    "department": obj.get("department"),
                    "additionalImages": obj.get("additionalImages"),
                    "artistDisplayBio": obj.get("artistDisplayBio"),
                },
            )
            results.append(rec)
    return results
