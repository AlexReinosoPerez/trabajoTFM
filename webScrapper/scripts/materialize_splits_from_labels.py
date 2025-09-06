# scripts/materialize_splits_from_labels.py
import os, shutil
from pathlib import Path
import pandas as pd

CSV = Path("data/metadata/labels.csv")
OUT = Path("data")

CLASS_MAP = {0: "not_vangogh", 1: "vangogh"}

def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        # hardlink (misma unidad de disco). Ahorra espacio.
        if dst.exists():
            return
        os.link(src, dst)
    except Exception:
        # fallback: copia
        shutil.copy2(src, dst)

def main():
    df = pd.read_csv(CSV)
    # esperamos columnas: path, label, split
    assert {"path","label","split"}.issubset(df.columns), "labels.csv debe tener path,label,split"

    n = 0
    for _, row in df.iterrows():
        src = Path(row["path"])
        cls = CLASS_MAP[int(row["label"])]
        split = row["split"]
        dst = OUT / split / cls / src.name
        safe_link_or_copy(src, dst)
        n += 1
    print(f"OK: materializados {n} archivos en {OUT}/(train|val|test)/(vangogh|not_vangogh)")

if __name__ == "__main__":
    main()
