import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def split_and_prepare_dataset(
    source_train_dir: str,
    source_test_dir: str,
    output_dir: str = "data",
    val_ratio: float = 0.2,
    seed: int = 42
):
    random.seed(seed)

    # Preparar carpetas de destino
    for split in ["train", "val"]:
        for label in ["Verdadera", "Falsa"]:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

    def get_class_from_filename(filename):
        if filename.lower().startswith("vg"):
            return "Verdadera"
        elif filename.lower().startswith("nvg"):
            return "Falsa"
        else:
            return None

    # 1. Procesar conjunto de entrenamiento
    from glob import glob

    # Obtener imágenes recursivamente
    all_train_images = glob(os.path.join(source_train_dir, "**", "*.*"), recursive=True)
    all_train_images = [f for f in all_train_images if get_class_from_filename(os.path.basename(f)) is not None]

    random.shuffle(all_train_images)

    val_count = int(len(all_train_images) * val_ratio)
    val_images = all_train_images[:val_count]
    train_images = all_train_images[val_count:]

    for image_set, split_name in zip([train_images, val_images], ["train", "val"]):
        for src in tqdm(image_set, desc=f"Copiando {split_name}"):
            fname = os.path.basename(src)
            cls = get_class_from_filename(fname)
            dst = os.path.join(output_dir, split_name, cls, fname)
            shutil.copy2(src, dst)

    # Procesar test igual
    test_images = glob(os.path.join(source_test_dir, "**", "*.*"), recursive=True)
    test_images = [f for f in test_images if get_class_from_filename(os.path.basename(f)) is not None]

    for src in tqdm(test_images, desc="Copiando test a val"):
        fname = os.path.basename(src)
        cls = get_class_from_filename(fname)
        dst = os.path.join(output_dir, "val", cls, fname)
        shutil.copy2(src, dst)


    print(f"\n✅ Dataset procesado correctamente en: {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepara el dataset VGDB-2016 para clasificación binaria.")
    parser.add_argument("--source_train_dir", type=str, required=True, help="Ruta a la carpeta 'train/' del dataset.")
    parser.add_argument("--source_test_dir", type=str, required=True, help="Ruta a la carpeta 'test/' del dataset.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directorio destino. Por defecto: ./data/")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Porcentaje para validación. Por defecto: 0.2")

    args = parser.parse_args()
    split_and_prepare_dataset(
        source_train_dir=args.source_train_dir,
        source_test_dir=args.source_test_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio
    )
