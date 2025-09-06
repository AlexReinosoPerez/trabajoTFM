#!/usr/bin/env python3
"""
Crea labels.csv uniendo:
 - Positivos (Van Gogh): una o varias carpetas
 - Negativos (no-VG): data/negatives (salida del script anterior)

Hace split estratificado train/val/test (70/15/15 por defecto).
"""

import os, argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_EXTS = (".jpg",".jpeg",".png",".tif",".tiff",".webp",".bmp")

def list_images(dirs):
    out=[]
    for d in dirs:
        p=Path(d)
        if not p.exists(): continue
        for ext in IMG_EXTS:
            out += [str(x) for x in p.rglob(f"*{ext}")]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-dirs", nargs="+", required=True, help="Carpetas con Van Gogh (877)")
    ap.add_argument("--neg-dir", required=True, help="Carpeta con negativos exportados")
    ap.add_argument("--out", default="data/metadata/labels.csv")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    args = ap.parse_args()

    pos = list_images(args.pos_dirs)
    neg = list_images([args.neg_dir])

    dfp = pd.DataFrame({"path":pos, "label":1})
    dfn = pd.DataFrame({"path":neg, "label":0})
    df = pd.concat([dfp, dfn], ignore_index=True)
    df["id"] = df["path"].apply(lambda s: os.path.splitext(os.path.basename(s))[0])

    # split
    train_size = args.train
    val_size = args.val
    test_size = 1.0 - train_size - val_size

    # estratificado por label
    df_train, df_tmp = train_test_split(df, test_size=(1-train_size), stratify=df["label"], random_state=42)
    df_val, df_test = train_test_split(df_tmp, test_size=test_size/(test_size+val_size),
                                       stratify=df_tmp["label"], random_state=42)

    for split, d in [("train",df_train),("val",df_val),("test",df_test)]:
        d = d.copy(); d["split"]=split
        if split=="train": out_df = d
        else: out_df = pd.concat([out_df,d], ignore_index=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df[["path","label","split"]].to_csv(args.out, index=False)
    print("[OK] labels.csv ->", args.out)
    print(out_df["split"].value_counts(), "\n", out_df["label"].value_counts())

if __name__ == "__main__":
    main()
