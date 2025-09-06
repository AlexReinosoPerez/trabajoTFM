#!/usr/bin/env python3
import os, argparse, time, json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import timm

# -------- Dataset --------
class VangoghDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size=224, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
        if augment:
            self.tf = T.Compose([
                T.Resize(int(img_size*1.15)),
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.1,0.1,0.1,0.05)], p=0.3),
                T.RandomApply([T.GaussianBlur(3)], p=0.2),
                T.ToTensor(),
                T.Normalize(mean,std)
            ])
        else:
            self.tf = T.Compose([
                T.Resize(int(img_size*1.10)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean,std)
            ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = row["path"]
        label = int(row["label"])
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
        except Exception:
            im = Image.new("RGB",(256,256),(0,0,0))
        x = self.tf(im)
        return x, torch.tensor(label, dtype=torch.long)

# -------- Utils --------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_loaders(labels_csv, img_size, batch, num_workers=4, weighted_sampler=True):
    df = pd.read_csv(labels_csv)
    df_train = df[df["split"]=="train"].copy()
    df_val   = df[df["split"]=="val"].copy()
    df_test  = df[df["split"]=="test"].copy()

    ds_train = VangoghDataset(df_train, img_size=img_size, augment=True)
    ds_val   = VangoghDataset(df_val,   img_size=img_size, augment=False)
    ds_test  = VangoghDataset(df_test,  img_size=img_size, augment=False)

    if weighted_sampler:
        counts = df_train["label"].value_counts().to_dict()
        # weights inversos por clase
        class_weights = {c: len(df_train)/counts[c] for c in counts}
        sample_weights = df_train["label"].map(class_weights).values
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(ds_train, batch_size=batch, sampler=sampler, num_workers=num_workers, pin_memory=True)
        cw_tensor = torch.tensor([class_weights.get(0,1.0), class_weights.get(1,1.0)], dtype=torch.float32)
    else:
        train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True)
        cw_tensor = None

    val_loader  = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, cw_tensor, (df_train, df_val, df_test)

def build_model(model_name="vit_base_patch16_224", num_classes=2, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:,1]
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# -------- Train --------
def train(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader, test_loader, cw_tensor, _ = make_loaders(
        labels_csv=args.labels, img_size=args.img_size, batch=args.batch, num_workers=args.workers, weighted_sampler=True
    )

    model = build_model(args.model, num_classes=2, pretrained=True).to(device)
    if args.freeze_backbone:
        for n,p in model.named_parameters():
            if "head" in n or "fc" in n or "classifier" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    # criterio y optim
    if cw_tensor is not None:
        cw_tensor = cw_tensor.to(device)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = -1.0
    best_path = os.path.join(args.out_dir, f"{args.model}_best.pth")
    patience = args.patience
    no_improve = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()*x.size(0)
        scheduler.step()
        train_loss = running/len(train_loader.dataset)

        yv, pv, _ = evaluate(model, val_loader, device)
        from sklearn.metrics import f1_score, accuracy_score
        val_acc = accuracy_score(yv, pv)
        val_f1  = f1_score(yv, pv)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            no_improve = 0
            print(f"  -> best so far. saved to {best_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Eval test con mejor checkpoint
    model = build_model(args.model, num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    yt, pt, prob = evaluate(model, test_loader, device)
    report = classification_report(yt, pt, target_names=["non_vg","van_gogh"], digits=4, output_dict=True)
    cm = confusion_matrix(yt, pt).tolist()

    # Guardar métricas
    metrics_path = os.path.join(args.out_dir, f"{args.model}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"report": report, "confusion_matrix": cm}, f, indent=2)
    print(f"[DONE] Test report saved to {metrics_path}")
    print(pd.DataFrame(report).transpose())
    print("Confusion matrix:", cm)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/metadata/labels.csv")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model", default="vit_base_patch16_224", help="e.g. vit_base_patch16_224 or resnet50")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--patience", type=int, default=4)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
