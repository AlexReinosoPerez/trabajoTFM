#!/usr/bin/env python3
"""
Fetch a large, license-clean dataset of authenticated Van Gogh works
from multiple open-access sources (Met, AIC, CMA, Rijksmuseum),
and save unified metadata (CSV + JSONL). Optionally download images.

Uso:
  python -m scripts.fetch_vangogh_dataset --sources met aic cma rijks --limit 500
  python -m scripts.fetch_vangogh_dataset --sources met aic --limit 200 --download
"""
import os
import argparse
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv

from vg_dataset_scraper.common import (
    save_jsonl,
    save_csv,
    build_session,
    download_image,
    ensure_dir,
)

# Importar scrapers
from vg_dataset_scraper.sources import met, aic, cma, rijks
from scripts import fetch_rijks_all

SOURCE_MAP = {
    "met": met.fetch,
    "aic": aic.fetch,
    "cma": cma.fetch,
    "rijks": rijks.fetch,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sources",
        nargs="+",
        default=["met", "aic", "cma", "rijks"],
        help="Fuentes a scrapear",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Máximo de registros por fuente",
    )
    p.add_argument(
        "--outdir",
        default="data",
        help="Directorio base de salida",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Descargar imágenes (puede ser pesado)",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=1_000_000_000,
        help="Límite máximo de imágenes a bajar (seguridad)",
    )
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    ensure_dir(args.outdir)
    all_rows: List[Dict] = []

    for src in args.sources:
        if src not in SOURCE_MAP:
            print(f"[WARN] Fuente desconocida: {src}")
            continue
        print(f"\n=== Fetching from {src} ===")
        try:
            rows = SOURCE_MAP[src](limit=args.limit)
            print(f"[OK] {src}: {len(rows)} registros")
            all_rows.extend(rows)
        except Exception as e:
            print(f"[ERROR] {src} falló: {e}. Continuando con las demás fuentes.")
            continue

    if not all_rows:
        print("[ERROR] No se obtuvieron registros de ninguna fuente.")
        return

    # Guardar metadatos unificados
    meta_dir = os.path.join(args.outdir, "metadata")
    ensure_dir(meta_dir)
    csv_path = os.path.join(meta_dir, "van_gogh_master.csv")
    jsonl_path = os.path.join(meta_dir, "van_gogh_master.jsonl")
    save_csv(csv_path, all_rows)
    save_jsonl(jsonl_path, all_rows)
    print(f"\n[INFO] Metadatos guardados en:\n  {csv_path}\n  {jsonl_path}")

    # Descarga de imágenes opcional
    if args.download:
        sess = build_session()
        img_base = os.path.join(args.outdir, "images")
        ensure_dir(img_base)
        downloaded = 0

        def iiif_from_manifest(manifest_url: str) -> str | None:
            try:
                m = sess.get(manifest_url, timeout=30)
                m.raise_for_status()
                mj = m.json()
            except Exception:
                return None

            # IIIF v2
            canvases = None
            sequences = (mj.get("sequences") or [])
            if sequences:
                canvases = sequences[0].get("canvases")

            # IIIF v3
            if not canvases:
                items = mj.get("items") or []
                if items and isinstance(items[0], dict):
                    canvases = items
            if not canvases:
                return None

            # elige primer canvas e intenta servicio IIIF
            c0 = canvases[0]
            svc_id = None
            direct_url = None

            images = c0.get("images") or []
            if images:
                res = images[0].get("resource") or {}
                svc = res.get("service") or {}
                svc_id = svc.get("@id") or svc.get("id")
                direct_url = res.get("@id") or res.get("id")

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
                return f"{svc_id}/full/2000,/0/default.jpg"
            return direct_url

        for row in tqdm(all_rows, desc="Descargando imágenes"):
            url = row.get("iiif_image_url")

            # NUEVO: si no hay url directa pero sí manifest, resuélvelo
            if not url:
                manifest = row.get("iiif_manifest_url")
                if manifest:
                    url = iiif_from_manifest(manifest)

            if not url:
                continue

            src = row["source"]
            obj_id = row["object_id"]
            ext = (
                ".jpg"
                if url.lower().endswith(".jpg") or ".jpeg" in url.lower()
                else (".tif" if ".tif" in url.lower() else ".bin")
            )
            out = os.path.join(img_base, src, f"{obj_id}{ext}")
            if os.path.exists(out):
                continue
            ok = download_image(sess, url, out)
            if ok:
                downloaded += 1
                if downloaded >= args.max_images:
                    break

        print(f"[INFO] Imágenes descargadas: {downloaded}")


if __name__ == "__main__":
    main()
