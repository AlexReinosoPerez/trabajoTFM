import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
import mlflow
import mlflow.pytorch

from utils.data import get_data_loaders
from utils.metrics import log_confusion_matrix, save_confusion_matrix_txt

# -----------------------------
# Configuración
# -----------------------------
DATA_DIR = "data"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_NAME = "models/resnet50_vangogh.pth"
EXPERIMENT_NAME = "VanGogh-TrueFalse"

# -----------------------------
# MLflow Setup
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# -----------------------------
# Carga de datos
# -----------------------------
train_loader, val_loader, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)

# -----------------------------
# Modelo
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 2)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=True)

# -----------------------------
# Entrenamiento
# -----------------------------
with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    mlflow.log_params({
        "architecture": "resnet50",
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW",
        "loss_fn": "CrossEntropyLoss"
    })

    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_preds, train_targets = 0.0, [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds)

        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds)
        scheduler.step(val_acc)

        mlflow.log_metrics({
            "train_loss": train_loss / len(train_loader.dataset),
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss / len(val_loader.dataset),
            "val_acc": val_acc,
            "val_f1": val_f1
        }, step=epoch)

        print(f"[{epoch+1}/{NUM_EPOCHS}] "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_NAME)
            mlflow.pytorch.log_model(model, artifact_path="model")

            cm = confusion_matrix(val_targets, val_preds)
            save_confusion_matrix_txt(cm, class_names)
            mlflow.log_artifact("confusion_matrix.txt")
            os.remove("confusion_matrix.txt")

    print("✅ Entrenamiento finalizado. Modelo y resultados registrados.")
