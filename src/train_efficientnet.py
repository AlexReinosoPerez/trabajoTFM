import os
import time
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from datetime import datetime
import mlflow
import mlflow.pytorch
from utils.data import get_data_loaders
from utils.metrics import log_confusion_matrix, save_confusion_matrix_txt
from PIL import Image
from utils.training import EarlyStopping

# -----------------------------
# Configuración
# -----------------------------
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../webScrapper/data'))
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10
MODEL_NAME = "models/efficientnetb0_vangogh.pth"
EXPERIMENT_NAME = "VanGogh-EfficientNetB0"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

Image.MAX_IMAGE_PIXELS = None
train_loader, val_loader, class_names, class_counts = get_data_loaders(DATA_DIR, BATCH_SIZE, use_sampler=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Inicializar EfficientNet-B0 con pesos preentrenados
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for param in efficientnet.parameters():
    param.requires_grad = True

# Sustituir la capa final
efficientnet.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(efficientnet.classifier[1].in_features, 2)
)
efficientnet = efficientnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(efficientnet.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    mlflow.log_params({
        "architecture": "efficientnet_b0",
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "loss_fn": "CrossEntropyLoss",
        "device": str(device),
        "finetuning": True
    })
    mlflow.log_dict(class_counts, "class_counts.json")

    best_acc = 0.0
    patience_counter = 0
    early_stopping = EarlyStopping(patience=3, verbose=True)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        efficientnet.train()
        train_loss, train_preds, train_targets = 0.0, [], []

        print(f"\n🔄 Epoch {epoch+1}/{NUM_EPOCHS} - Entrenando...")

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = efficientnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, zero_division=1)
        train_precision = precision_score(train_targets, train_preds, zero_division=1)
        train_recall = recall_score(train_targets, train_preds, zero_division=1)

        efficientnet.eval()
        val_loss, val_preds, val_targets = 0.0, [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = efficientnet(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=1)
        val_precision = precision_score(val_targets, val_preds, zero_division=1)
        val_recall = recall_score(val_targets, val_preds, zero_division=1)

        scheduler.step()

        mlflow.log_metrics({
            "train_loss": train_loss / len(train_loader.dataset),
            "train_acc": train_acc,
            "train_f1": train_f1,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "val_loss": val_loss / len(val_loader.dataset),
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall
        }, step=epoch)

        duration = time.time() - start_time
        print(f"🕒 Epoch {epoch+1} completada en {duration:.2f} segundos")
        print(f"📊 Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_NAME), exist_ok=True)
            torch.save(efficientnet.state_dict(), MODEL_NAME)
            # Guardar también en src/models si es el mejor
            os.makedirs("src/models", exist_ok=True)
            torch.save(efficientnet.state_dict(), "src/models/efficientnetb0_vangogh.pth")
            # mlflow.pytorch.log_model(efficientnet, artifact_path="model")

            cm = confusion_matrix(val_targets, val_preds)
            save_confusion_matrix_txt(cm, class_names)
            mlflow.log_artifact("confusion_matrix.txt")
            os.remove("confusion_matrix.txt")

        # ⏹️ Evaluar early stopping
        if early_stopping(val_acc):
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

    print("✅ Entrenamiento finalizado. Modelo y resultados registrados.")