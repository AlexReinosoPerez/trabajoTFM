"""
Rijksmuseum Linked Art Search API scraper for Vincent van Gogh.
- Paginates through the entire search (follows "next")
- Dereferences Linked Art JSON for each item
- Tries to resolve a IIIF image URL from the Presentation v2/v3 manifest
- Returns normalized records compatible with the project schema

Docs:
- Search API (Linked Art): https://data.rijksmuseum.nl/docs/search
- IIIF: https://data.rijksmuseum.nl/docs/iiif/
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import requests

from ..common import build_session, normalize_record

SEARCH_URL = "https://data.rijksmuseum.nl/search/collection"
LICENSE_SHORT = "Open data (image license varies)"
LICENSE_URL = "https://data.rijksmuseum.nl/"
TIMEOUT = 30


def _dereference_linked_art(sess: requests.Session, lod_id: str) -> Optional[Dict[str, Any]]:
    """Get Linked Art JSON via content negotiation; follow redirects."""
    try:
        r = sess.get(lod_id, headers={"Accept": "application/ld+json"}, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code == 406:
            r = sess.get(lod_id, timeout=TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _extract_core_fields(la: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract title, accession, date, medium, dimensions from Linked Art JSON."""
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


def _iiif_from_manifest(sess: requests.Session, manifest_url: str, width_pref: int = 2000) -> Optional[str]:
    """
    Parse IIIF Presentation v2/v3 manifest to build a large image URL.
    Prefers the IIIF Image API service when available; falls back to direct body URL.
    """
    if not manifest_url:
        return None
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

    # choose first canvas; build URL
    c0 = canvases[0]
    svc_id = None
    direct_url = None

    # v2: images[0].resource
    images = c0.get("images") or []
    if images:
        res = images[0].get("resource") or {}
        svc = res.get("service") or {}
        svc_id = svc.get("@id") or svc.get("id")
        direct_url = res.get("@id") or res.get("id")

    # v3: items[0].items[0].body
    if not direct_url:
        citems = c0.get("items") or []
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

    if svc_id:
        svc_id = svc_id.rstrip("/")
        return f"{svc_id}/full/{width_pref},/0/default.jpg"
    return direct_url


def fetch(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Fetch Van Gogh records from Rijksmuseum Linked Art API up to `limit`.
    Returns normalized dicts with as many iiif_image_url populated as possible.
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

            la = _dereference_linked_art(sess, lod_id)
            if not la:
                continue

            title, accession, date, medium, dimensions = _extract_core_fields(la)

            # Find IIIF manifest
            manifest_url = None
            for rep in (la.get("representation") or []):
                srv = rep.get("service")
                if isinstance(srv, dict) and "id" in srv and "iiif" in (srv.get("conforms_to") or ""):
                    manifest_url = srv.get("id")
                    break
                if rep.get("id", "").endswith("manifest"):
                    manifest_url = rep["id"]
                    break
            if not manifest_url:
                for s in (la.get("subject_of") or []):
                    if s.get("id", "").endswith("manifest"):
                        manifest_url = s["id"]
                        break

            # Map an image URL when possible
            iiif_image_url = _iiif_from_manifest(sess, manifest_url, width_pref=2000) if manifest_url else None

            oid = lod_id.split("/")[-1]
            rec = normalize_record(
                source="rijks",
                object_id=oid,
                title=title,
                date=date,
                medium=medium,
                dimensions=dimensions,
                artist="Vincent van Gogh",
                museum="Rijksmuseum",
                accession_number=accession,
                object_url=la.get("id"),
                iiif_image_url=iiif_image_url,
                iiif_manifest_url=manifest_url,
                license_short=LICENSE_SHORT,
                license_url=LICENSE_URL,
                extra={"linked_art": True},
            )
            results.append(rec)

        nxt = (data.get("next") or {}).get("id")
        if not nxt:
            break
        next_url = nxt
        params = None
        time.sleep(0.2)  # be gentle

    return results
