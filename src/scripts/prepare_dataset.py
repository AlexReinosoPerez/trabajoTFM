import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def prepare_dataset(
    original_dir: str,
    output_dir: str = "data",
    val_ratio: float = 0.2,
    seed: int = 42
):
    random.seed(seed)

    # Definimos mapeo de carpetas originales a clases
    class_map = {
        "vangogh": "Verdadera",
        "not_vangogh": "Falsa"
    }

    # Crear carpetas de destino
    for split in ["train", "val"]:
        for label in class_map.values():
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

    for original_class, target_class in class_map.items():
        source_folder = os.path.join(original_dir, original_class)
        all_images = [f for f in os.listdir(source_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(all_images)

        val_count = int(len(all_images) * val_ratio)
        val_images = all_images[:val_count]
        train_images = all_images[val_count:]

        # Copiar archivos
        for img in tqdm(train_images, desc=f"Copiando {target_class} - train"):
            src = os.path.join(source_folder, img)
            dst = os.path.join(output_dir, "train", target_class, img)
            shutil.copy2(src, dst)

        for img in tqdm(val_images, desc=f"Copiando {target_class} - val"):
            src = os.path.join(source_folder, img)
            dst = os.path.join(output_dir, "val", target_class, img)
            shutil.copy2(src, dst)

    print(f"\n✅ Dataset preparado correctamente en: {output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepara el dataset de Van Gogh vs No Van Gogh.")
    parser.add_argument("--original_dir", type=str, required=True, help="Ruta a la carpeta que contiene vangogh/ y not_vangogh/")
    parser.add_argument("--output_dir", type=str, default="data", help="Carpeta destino (por defecto: ./data/)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Porcentaje para validación (por defecto: 0.2)")

    args = parser.parse_args()
    prepare_dataset(args.original_dir, args.output_dir, args.val_ratio)
