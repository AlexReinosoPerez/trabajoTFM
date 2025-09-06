#!/usr/bin/env python3
import os, sys, pandas as pd

MASTER = os.path.join("data","metadata","van_gogh_master.csv")

def main():
    if len(sys.argv) < 2:
        print("Uso: python -m scripts.append_to_master --csv <ruta>")
        return
    if sys.argv[1] != "--csv" or len(sys.argv) < 3:
        print("Uso: python -m scripts.append_to_master --csv <ruta>")
        return
    extra = sys.argv[2]
    if not os.path.exists(extra):
        print(f"[ERROR] No existe: {extra}")
        return

    if os.path.exists(MASTER):
        dfm = pd.read_csv(MASTER)
    else:
        dfm = pd.DataFrame()

    dfx = pd.read_csv(extra)
    # concat columnas heterogéneas
    df_all = pd.concat([dfm, dfx], ignore_index=True, sort=False)

    # dedupe ligero por claves robustas
    key_cols = []
    if "accession_number" in df_all.columns: key_cols.append("accession_number")
    if "object_url" in df_all.columns:       key_cols.append("object_url")
    # fallback por (source, object_id)
    if "source" in df_all.columns and "object_id" in df_all.columns:
        df_all["_fallback"] = df_all["source"].astype(str)+"||"+df_all["object_id"].astype(str)
        key_cols.append("_fallback")

    if key_cols:
        df_all = df_all.drop_duplicates(subset=key_cols, keep="first")
        df_all = df_all.drop(columns=["_fallback"], errors="ignore")

    os.makedirs(os.path.dirname(MASTER), exist_ok=True)
    df_all.to_csv(MASTER, index=False)
    print(f"[OK] master actualizado -> {MASTER}")
    print(df_all["source"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
