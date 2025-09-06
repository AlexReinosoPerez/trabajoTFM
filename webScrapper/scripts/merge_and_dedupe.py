#!/usr/bin/env python3
"""
Merge and dedupe Van Gogh metadata collected across multiple runs/sources.
- Input: data/metadata/van_gogh_master.csv  (ya generado por tu scraper)
- Output: data/metadata/van_gogh_unique.csv (deduplicated)

Deduplication keys (in order of confidence):
1) accession_number (cuando existe)
2) object_url       (cuando existe)
3) (source, object_id) fallback

Además, marca si hay archivo de imagen descargado y su ruta.
"""

import os
import pandas as pd

IN_CSV = os.path.join("data", "metadata", "van_gogh_master.csv")
OUT_CSV = os.path.join("data", "metadata", "van_gogh_unique.csv")
IMAGES_DIR = os.path.join("data", "images")

def first_nonnull(s):
    for x in s:
        if pd.notna(x) and str(x).strip():
            return x
    return None

def main():
    if not os.path.exists(IN_CSV):
        print(f"[ERROR] Not found: {IN_CSV}")
        return
    df = pd.read_csv(IN_CSV)

    # normalizar strings basicas
    for col in ["accession_number","object_url","source","object_id","iiif_image_url","iiif_manifest_url","title"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": None, "None": None, "": None})

    # generar una clave auxiliar de fallback
    df["_fallback_key"] = (df.get("source", "") + "||" + df.get("object_id", "")).astype(str)

    # orden de preferencia para dedupe
    def make_key(row):
        if row.get("accession_number"):
            return ("ACC", row["accession_number"])
        if row.get("object_url"):
            return ("URL", row["object_url"])
        return ("FALL", row["_fallback_key"])

    keys = df.apply(make_key, axis=1)
    df["_dedupe_key"] = keys

    # agrupar y elegir la "mejor" fila por grupo (preferimos con iiif_image_url)
    groups = []
    for k, g in df.groupby("_dedupe_key", dropna=False):
        # priorizar fila con imagen directa; si no hay, aceptamos cualquiera
        g = g.copy()
        has_img = g["iiif_image_url"].notna()
        if has_img.any():
            chosen = g.loc[has_img].iloc[0]
        else:
            chosen = g.iloc[0]
        groups.append(chosen)

    out = pd.DataFrame(groups).drop(columns=["_fallback_key","_dedupe_key"], errors="ignore")

    # añadir columna filepath si existe imagen descargada
    paths = []
    for _, r in out.iterrows():
        src = r.get("source")
        oid = r.get("object_id")
        if not src or not oid:
            paths.append(None)
            continue
        # probar extensiones comunes
        base = os.path.join(IMAGES_DIR, str(src), str(oid))
        for ext in (".jpg",".jpeg",".tif",".tiff",".png",".bin"):
            p = base + ext
            if os.path.exists(p):
                paths.append(p)
                break
        else:
            paths.append(None)
    out["local_image_path"] = paths

    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote deduped file: {OUT_CSV}")
    print(f"     Unique rows: {len(out)}")

if __name__ == "__main__":
    main()
