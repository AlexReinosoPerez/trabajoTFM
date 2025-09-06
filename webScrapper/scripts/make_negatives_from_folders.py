#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construye negativos (no-Van Gogh) desde:
  - una o varias carpetas de candidatos (postimpresionismo, árbol de artistas)
  - referencias de Van Gogh (tus 877)

Filtrado:
  1) Exclusión por ruta (carpetas/nombres que contengan 'van gogh', 'vincent_van_gogh', etc.)
  2) pHash near-duplicate contra referencias (captura copias/recortes)
  3) CLIP: similitud con imágenes de referencia y con texto 'a painting by Vincent van Gogh'
     -> si supera umbral => sospechoso VG => se descarta del set negativo

Selección:
  - Mantiene los NO-VG y prioriza 'hard negatives' (scores cercanos al umbral pero por debajo)
  - target número total (p.ej. 1500)

Salida:
  - Copia en data/negatives/
  - data/metadata/negatives_manifest.csv con métricas y flags
"""

import os, re, sys, argparse, shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp")


# ---------- utils ----------
def list_images(root_dirs: List[str]) -> List[Path]:
    out: List[Path] = []
    for rd in root_dirs:
        p = Path(rd)
        if not p.exists():
            continue
        for ext in IMG_EXTS:
            out.extend(p.rglob(f"*{ext}"))
    # únicos
    seen = set()
    uniq: List[Path] = []
    for x in out:
        s = str(x)
        if s not in seen:
            seen.add(s)
            uniq.append(x)
    return uniq


def safe_open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def compute_phash(path: Path) -> int:
    try:
        im = safe_open_rgb(path)
        # Devolvemos entero Python (sin numpy) para evitar overflow
        return int(str(imagehash.phash(im)), 16)
    except Exception:
        return -1


def hamming(a: int, b: int) -> int:
    if a == -1 or b == -1:
        return 64
    return (a ^ b).bit_count()


def looks_like_vangogh_path(p: Path) -> bool:
    s = str(p).lower().replace("-", "_").replace(" ", "_")
    patterns = [
        r"vincent[_]?van[_]?gogh",
        r"\bvan[_]?gogh\b",
        r"\bvincent\b.*\bgogh\b",
    ]
    return any(re.search(pt, s) for pt in patterns)


def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device


@torch.no_grad()
def embed_images(model, processor, device, paths: List[Path], batch: int = 16) -> Tuple[np.ndarray, List[str]]:
    embs = []
    good: List[str] = []
    if not paths:
        return np.zeros((0, 512), dtype=np.float32), []
    for i in tqdm(range(0, len(paths), batch), desc="Embedding images"):
        chunk = paths[i : i + batch]
        imgs = []
        for p in chunk:
            try:
                imgs.append(safe_open_rgb(p))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        embs.append(feats.cpu().numpy())
        good += [str(p) for p in chunk]
    return np.vstack(embs).astype(np.float32), good


@torch.no_grad()
def embed_text(model, processor, device, text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt").to(device)
    feats = model.get_text_features(**inputs)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats[0].cpu().numpy().astype(np.float32)


def cosine_max(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (max_sim_por_fila, idx_col_correspondiente).
    Si b está vacío, devuelve arrays de ceros del tamaño de a.
    """
    if a.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if b.size == 0:
        return np.zeros((a.shape[0],), dtype=np.float32), np.zeros((a.shape[0],), dtype=np.int64)
    sims = a @ b.T
    idx = np.argmax(sims, axis=1)
    mx = sims[np.arange(len(a)), idx]
    return mx.astype(np.float32), idx.astype(np.int64)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate-dirs",
        nargs="+",
        required=True,
        help="Carpetas con candidatos negativos (postimpresionismo, árbol artistas)",
    )
    ap.add_argument(
        "--vg-ref-dirs",
        nargs="+",
        required=True,
        help="Carpetas con positivos de Van Gogh (tus 877)",
    )
    ap.add_argument("--out-dir", default="data/negatives", help="Salida (copias) de negativos")
    ap.add_argument("--manifest", default="data/metadata/negatives_manifest.csv")
    ap.add_argument("--target", type=int, default=1500, help="Número objetivo de negativos")
    ap.add_argument("--threshold", type=float, default=0.30, help="Umbral combinado para marcar 'sospechoso VG'")
    ap.add_argument("--w-ref", type=float, default=0.6, help="Peso similitud vs referencias")
    ap.add_argument("--w-text", type=float, default=0.4, help="Peso similitud vs texto")
    ap.add_argument("--phash-hd", type=int, default=10, help="Hamming pHash <= HD => near-duplicate VG")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hard-ratio", type=float, default=0.6, help="Proporción de 'hard negatives' (cercanos al umbral)")
    ap.add_argument("--copy", action="store_true", help="Copia archivos (por defecto copia).")
    ap.add_argument("--move", action="store_true", help="Mueve archivos en lugar de copiar.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Cargar referencias (VG)
    print("[INFO] Listando referencias (VG)...")
    ref_paths = list_images(args.vg_ref_dirs)
    print(f"[INFO] Referencias VG: {len(ref_paths)} imágenes")
    if len(ref_paths) == 0:
        print("[ERROR] No se encontraron imágenes de referencia (Van Gogh). Revisa --vg-ref-dirs.")
        sys.exit(1)

    ref_phashes = [compute_phash(Path(p)) for p in tqdm(ref_paths, desc="pHash refs")]
    model, processor, device = load_clip()
    ref_embs, ref_paths_ok = embed_images(model, processor, device, [Path(p) for p in ref_paths], batch=args.batch)
    # realinear pHash al orden ok → lista de enteros Python (sin overflow)
    p2h: Dict[str, int] = {str(p): h for p, h in zip(ref_paths, ref_phashes)}
    ref_ph_ok: List[int] = [p2h.get(p, -1) for p in ref_paths_ok]

    text_emb = embed_text(model, processor, device, "a painting by Vincent van Gogh")
    # normalizado ya por CLIP, pero aseguramos norma
    nrm = np.linalg.norm(text_emb)
    if nrm > 0:
        text_emb = text_emb / nrm

    # 2) Cargar candidatos y filtrar por ruta (excluir obvios VG)
    print("[INFO] Listando candidatos (posibles negativos)...")
    cand_paths_all = list_images(args.candidate_dirs)
    if len(cand_paths_all) == 0:
        print("[ERROR] No se encontraron candidatos en --candidate-dirs.")
        sys.exit(1)

    cand_paths = [p for p in cand_paths_all if not looks_like_vangogh_path(Path(p))]
    print(f"[INFO] Candidatos tras exclusión por ruta: {len(cand_paths)} / {len(cand_paths_all)}")
    if len(cand_paths) == 0:
        print("[ERROR] Tras excluir rutas con 'van gogh', no quedan candidatos. Revisa las carpetas.")
        sys.exit(1)

    # 3) pHash candidatos (lista de ints Python)
    cand_ph_list: List[int] = [compute_phash(Path(p)) for p in tqdm(cand_paths, desc="pHash candidates")]

    # 4) Embeddings candidatos (CLIP)
    cand_embs, cand_paths_ok = embed_images(model, processor, device, [Path(p) for p in cand_paths], batch=args.batch)
    if cand_embs.size == 0 or len(cand_paths_ok) == 0:
        print("[ERROR] No se pudieron embedir candidatos (cand_embs vacío).")
        sys.exit(1)

    # realinear pHash a paths_ok (lista de ints, sin numpy)
    p2h_c: Dict[str, int] = {str(p): h for p, h in zip(cand_paths, cand_ph_list)}
    cand_ph_ok: List[int] = [p2h_c.get(p, -1) for p in cand_paths_ok]

    # 5) Similitudes
    sim_ref_max, ref_idx = cosine_max(cand_embs, ref_embs)  # (N,) y (N,)
    sim_text = (cand_embs @ text_emb.reshape(-1, 1)).squeeze(1)  # (N,)
    score = args.w_ref * sim_ref_max + args.w_text * sim_text    # (N,)

    # 6) Near-duplicate por pHash (O(N*M); OK para ~7.5k x 877)
    print("[INFO] Calculando near-duplicates pHash (esto puede tardar un poco)...")
    phash_hit_list: List[bool] = []
    for hp in tqdm(cand_ph_ok, desc="pHash compare"):
        hd_min = 64
        for rh in ref_ph_ok:
            d = hamming(hp, rh)
            if d < hd_min:
                hd_min = d
            if hd_min <= args.phash_hd:
                break
        phash_hit_list.append(hd_min <= args.phash_hd)
    phash_hit = np.array(phash_hit_list, dtype=bool)

    # 7) Decisión: sospechoso VG si pHash_hit OR score >= threshold
    suspect_vg = phash_hit | (score >= args.threshold)
    keep_neg = ~suspect_vg

    # construir dataframe
    rows: List[Dict[str, Any]] = []
    for p, s_ref, s_txt, sc, phit, sus in zip(
        cand_paths_ok, sim_ref_max, sim_text, score, phash_hit, suspect_vg
    ):
        rows.append(
            {
                "path": p,
                "sim_ref_max": float(s_ref),
                "sim_text": float(s_txt),
                "score_combined": float(sc),
                "phash_near_dup": bool(phit),
                "suspect_vg": bool(sus),
            }
        )
    df = pd.DataFrame(rows)

    df_keep = df[~df["suspect_vg"]].copy()
    print(f"[OK] Negativos candidatos (NO-VG): {len(df_keep)}")
    if len(df_keep) == 0:
        print("[ERROR] Todos los candidatos fueron marcados como sospechosos de VG. "
              "Prueba bajando --threshold (p.ej. 0.28) o revisa las carpetas.")
        sys.exit(1)

    # 8) Priorización hard-negatives (más cercanos al umbral por debajo)
    df_keep["dist_to_thr"] = (args.threshold - df_keep["score_combined"]).abs()
    df_keep = df_keep.sort_values(by=["dist_to_thr"], ascending=True)

    n_target = min(args.target, len(df_keep))
    n_hard = int(n_target * args.hard_ratio)
    hard = df_keep.iloc[:n_hard]
    rest = (
        df_keep.iloc[n_hard:].sample(n_target - n_hard, random_state=42)
        if n_target > n_hard
        else pd.DataFrame()
    )
    df_sel = pd.concat([hard, rest], ignore_index=True)

    # 9) Copiado/movido a out-dir
    op = shutil.move if args.move else shutil.copy2
    exported: List[str] = []
    for src in tqdm(df_sel["path"].tolist(), desc="Export negatives"):
        dst = os.path.join(args.out_dir, os.path.basename(src))
        if not os.path.exists(dst):
            try:
                op(src, dst)
            except Exception:
                continue
        exported.append(dst)

    df_sel = df_sel.assign(path_exported=exported[: len(df_sel)])
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    df_sel.to_csv(args.manifest, index=False)

    # Estadísticas finales
    n_sus = int(df["suspect_vg"].sum())
    n_keep = int((~df["suspect_vg"]).sum())
    print("\n========== RESUMEN ==========")
    print(f"Total candidatos     : {len(df)}")
    print(f"Sospechosos VG       : {n_sus}")
    print(f"NO-VG (candidatos)   : {n_keep}")
    print(f"Exportados (negativos): {len(exported)} -> {args.out_dir}")
    print(f"Manifiesto           : {args.manifest}")
    print("=============================\n")
    print("[DONE] Proceso completado.")

if __name__ == "__main__":
    main()
