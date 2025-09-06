"""
Art Institute of Chicago API. Docs: https://api.artic.edu/ (Public API + IIIF)
Descarga obras de "Vincent van Gogh" con control de ritmo y FALLBACK de consulta.

ENV (opcional):
- AIC_START_PAGE        (int, por defecto 1)    -> página inicial
- AIC_MAX_PAGES         (int, por defecto None) -> nº de páginas a recorrer en esta tanda
- AIC_PER_PAGE_LIMIT    (int, por defecto 50)   -> items por página (<=100)
- AIC_PER_PAGE_SLEEP    (float, por defecto 1.6)-> pausa base entre páginas (s)
- AIC_403_RETRIES       (int, por defecto 8)    -> reintentos cuando 403
- AIC_BACKOFF_INITIAL   (float, por defecto 8.0)-> backoff inicial (s) en 403
- AIC_JITTER_MAX        (float, por defecto 0.9)-> jitter aleatorio extra (s)
"""

from typing import List, Dict, Any
import os, time, random
import requests
from ..common import build_session, normalize_record

SEARCH_URL = "https://api.artic.edu/api/v1/artworks/search"
FIELDS = ",".join([
    "id","title","artist_title","date_display","medium_display","dimensions",
    "credit_line","image_id","iiif_url","is_public_domain","api_link","department_title"
])
LICENSE_SHORT = "Public domain (AIC); data under AIC terms"
LICENSE_URL = "https://www.artic.edu/open-access/public-api"

# Defaults (sobrescribibles por ENV)
START_PAGE       = int(os.getenv("AIC_START_PAGE", "1"))
_MAX_PAGES       = os.getenv("AIC_MAX_PAGES")
MAX_PAGES        = int(_MAX_PAGES) if (_MAX_PAGES and _MAX_PAGES.isdigit()) else None
PER_PAGE_LIMIT   = max(1, min(int(os.getenv("AIC_PER_PAGE_LIMIT", "50")), 100))
PER_PAGE_SLEEP   = float(os.getenv("AIC_PER_PAGE_SLEEP", "1.6"))
MAX_403_RETRIES  = int(os.getenv("AIC_403_RETRIES", "8"))
BACKOFF_INITIAL  = float(os.getenv("AIC_BACKOFF_INITIAL", "8.0"))
JITTER_MAX       = float(os.getenv("AIC_JITTER_MAX", "0.9"))

QUERY_TRIES = [
    {"q": "Vincent van Gogh"},
    {"q": "\"Vincent van Gogh\""},
    {"q": "artist_title:\"Vincent van Gogh\""},
]

def _page_request(sess: requests.Session, params_base: Dict[str, Any]):
    """Intenta la página con varias consultas fallback hasta devolver data no vacía o agotar opciones."""
    backoff = BACKOFF_INITIAL
    for attempt in range(MAX_403_RETRIES + 1):
        for qparams in QUERY_TRIES:
            params = params_base.copy()
            params.update(qparams)
            try:
                r = sess.get(SEARCH_URL, params=params, timeout=30)
                if r.status_code == 403:
                    # Rate limit: backoff y reintento de TODO el bloque
                    break
                r.raise_for_status()
                data = r.json()
                arr = data.get("data", []) or []
                if arr:
                    return data, arr
                # si vacía, probar siguiente variante de query en la misma página
            except requests.RequestException:
                # red issue: probar siguiente variante o reintentar ciclo
                continue
        # si llegamos aquí, o 403 o todas las queries vacías -> backoff si 403, si no, devolvemos vacío
        if r.status_code == 403:
            sleep_s = backoff + random.uniform(0, JITTER_MAX)
            time.sleep(sleep_s)
            backoff *= 1.7
            continue
        # no fue 403 y no hay datos: devolvemos vacío para que el caller avance de página
        return {"pagination": {}}, []
    # agotado por 403 continuos
    return {"pagination": {}}, []

def fetch(limit: int = 1000) -> List[Dict[str, Any]]:
    sess = build_session()
    sess.headers.update({
        "Referer": "https://www.artic.edu/",
        "User-Agent": "aic-scraper/1.0 (+research/educational use)",
    })

    results: List[Dict[str, Any]] = []
    page = int(os.getenv("AIC_START_PAGE", "1"))
    max_pages_env = os.getenv("AIC_MAX_PAGES")
    max_pages = int(max_pages_env) if (max_pages_env and max_pages_env.isdigit()) else None
    per_page = max(1, min(int(os.getenv("AIC_PER_PAGE_LIMIT", "50")), 100))
    sleep_s = float(os.getenv("AIC_PER_PAGE_SLEEP", "1.6"))
    max_403 = int(os.getenv("AIC_403_RETRIES", "8"))
    backoff0 = float(os.getenv("AIC_BACKOFF_INITIAL", "8.0"))
    jitter = float(os.getenv("AIC_JITTER_MAX", "0.9"))

    pages_done = 0
    while True:
        if max_pages is not None and pages_done >= max_pages:
            break
        if len(results) >= limit:
            break

        # Usa EXACTAMENTE los mismos params que probaste desde PowerShell:
        params = {
            "q": "Vincent van Gogh",
            "query[term][is_public_domain]": "true",
            "fields": "id,title,artist_title,date_display,medium_display,dimensions,credit_line,image_id,iiif_url,is_public_domain,api_link,department_title",
            "limit": min(per_page, limit - len(results)),
            "page": page,
        }

        # Backoff en 403
        backoff = backoff0
        for attempt in range(max_403 + 1):
            try:
                r = sess.get(SEARCH_URL, params=params, timeout=30)
                if r.status_code == 403:
                    if attempt >= max_403:
                        return results
                    time.sleep(backoff + random.uniform(0, jitter))
                    backoff *= 1.7
                    continue
                r.raise_for_status()
                break
            except requests.HTTPError:
                return results
            except requests.RequestException:
                time.sleep(3)
                continue

        data = r.json()
        arr = data.get("data") or []
        for it in arr:
            # Asegura PD + imagen IIIF
            if not it.get("is_public_domain"):
                continue
            if not it.get("image_id") or not it.get("iiif_url"):
                continue

            iiif = f"{it['iiif_url'].rstrip('/')}/{it['image_id']}/full/2000,/0/default.jpg"
            rec = normalize_record(
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
            results.append(rec)
            if len(results) >= limit:
                break

        # avanzar página
        pages_done += 1
        page += 1
        time.sleep(sleep_s + random.uniform(0, jitter))

        info = data.get("pagination") or data.get("info", {}).get("pagination")
        if info and page > info.get("total_pages", page):
            break

    return results[:limit]
    sess = build_session()
    sess.headers.update({
        "Referer": "https://www.artic.edu/",
        "User-Agent": "aic-scraper/1.0 (+research/educational use)",
    })

    results: List[Dict[str, Any]] = []
    page = START_PAGE
    pages_done = 0

    while True:
        if MAX_PAGES is not None and pages_done >= MAX_PAGES:
            break
        if len(results) >= limit:
            break

        per_page = min(PER_PAGE_LIMIT, limit - len(results))
        base_params = {
            "fields": FIELDS,
            "limit": per_page,
            "page": page,
        }

        data, arr = _page_request(sess, base_params)
        if not arr:
            # nada en esta página con ninguno de los queries -> pasar a la siguiente
            time.sleep(PER_PAGE_SLEEP + random.uniform(0, JITTER_MAX))
            page += 1
            pages_done += 1
            # si paginación oficial dice fin, cortamos
            info = data.get("pagination") or data.get("info", {}).get("pagination")
            if info and page > info.get("total_pages", page):
                break
            continue

        for it in arr:
            if not it.get("is_public_domain"):
                continue
            if not it.get("image_id") or not it.get("iiif_url"):
                continue
            iiif = f"{it['iiif_url'].rstrip('/')}/{it['image_id']}/full/2000,/0/default.jpg"
            rec = normalize_record(
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
            results.append(rec)
            if len(results) >= limit:
                break

        # pausa entre páginas con jitter
        time.sleep(PER_PAGE_SLEEP + random.uniform(0, JITTER_MAX))
        page += 1
        pages_done += 1

        info = data.get("pagination") or data.get("info", {}).get("pagination")
        if info and page > info.get("total_pages", page):
            break

    return results[:limit]
