from pathlib import Path
import shutil, time, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

REL_DATA = "data/modified"
SPLIT = "test"
CKPT_PATH = Path("checkpoints_vit_b16/vit_b16_best.pt")
OUT_DIR = Path("eval_vit_b16")
BATCH_SIZE = 64
NUM_WORKERS = 4

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    print("[WARN] No GPU found. Falling back to CPU."); return torch.device("cpu")

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        path = self.samples[index][0]
        return img, target, path

def build_model(num_classes: int, img_size: int):
    m = models.vit_b_16(weights=None, image_size=img_size)
    in_f = m.heads.head.in_features
    m.heads.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, num_classes))
    return m

@torch.no_grad()
def run_eval(model, dl, device):
    model.eval()
    y_true, y_pred, paths = [], [], []
    for x, y, p in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type!="cpu" else torch.bfloat16, enabled=(device.type!="cpu")):
            logits = model(x)
        pred = logits.argmax(1)
        y_true.extend(y.tolist()); y_pred.extend(pred.tolist()); paths.extend(list(p))
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    acc = (y_true == y_pred).mean()
    return acc, y_true, y_pred, paths

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def main():
    device = pick_device()
    print(f"[INFO] Device: {device}")

    if not CKPT_PATH.exists():
        raise SystemExit(f"[ERROR] Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    classes = ckpt.get("classes", None)
    if classes is None:
        raise SystemExit("[ERROR] 'classes' not found in checkpoint")
    img_size = ckpt.get("config", {}).get("img_size", 224)

    eval_tfms = transforms.Compose([
        transforms.CenterCrop(img_size),  # no-op если уже img_size×img_size
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    data_root = (Path(__file__).resolve().parent / REL_DATA).resolve()
    ds_test = ImageFolderWithPaths(data_root / SPLIT, transform=eval_tfms)
    if len(ds_test.classes) != len(classes):
        print(f"[WARN] Class count mismatch: dataset={len(ds_test.classes)} vs ckpt={len(classes)}")
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    model = build_model(len(classes), img_size).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    acc, y_true, y_pred, paths = run_eval(model, dl_test, device)
    dt = time.perf_counter() - t0

    print(f"[INFO] Test accuracy: {acc:.4f} ({(y_true==y_pred).sum()}/{len(y_true)})")
    print(f"[INFO] Evaluated in {dt:.2f}s")

    cm = confusion_matrix(y_true, y_pred, len(classes))
    import pandas as pd
    df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
    xlsx_path = OUT_DIR / "confusion_matrix.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="confusion_matrix")
    print(f"[INFO] Saved confusion matrix: {xlsx_path}")

    mis_dir = OUT_DIR / "misclassified"
    mis_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t, p, src in zip(y_true, y_pred, paths):
        if t != p:
            t_name, p_name = classes[t], classes[p]
            sub = mis_dir / f"true_{t_name}__pred_{p_name}"
            sub.mkdir(parents=True, exist_ok=True)
            dst = sub / Path(src).name
            try:
                if not dst.exists():
                    shutil.copy2(src, dst)
            except Exception as e:
                print(f"[WARN] Failed to copy {src}: {e}")
            rows.append({"path": src, "true": t_name, "pred": p_name})

    mis_csv = OUT_DIR / "misclassified.csv"
    pd.DataFrame(rows).to_csv(mis_csv, index=False)
    print(f"[INFO] Saved misclassified images to: {mis_dir}")
    print(f"[INFO] Saved misclassified list: {mis_csv}")

if __name__ == "__main__":
    main()
