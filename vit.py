from pathlib import Path
import time, json, random, csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

REL_DATA = "data/modified"
OUT_DIR = Path("checkpoints_vit_b16")
BATCH_SIZE = 64
EPOCHS = 60
BASE_LR = 1e-4
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 5e-2
NUM_WORKERS = 6
IMG_SIZE = 224
LABEL_SMOOTH = 0.05
SEED = 42
LOG_CSV = "train_log.csv"
PATIENCE = 8
MIN_DELTA = 1e-4
LINEAR_PROBE_EPOCHS = 2

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    print("[WARN] No GPU found. Falling back to CPU."); return torch.device("cpu")

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def init_csv_log(path: Path):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch","phase","train_loss","train_acc","val_loss","val_acc","lr","epoch_time_sec"])

def build_dataloaders(root: Path):
    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR)], p=0.3),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds_tr = datasets.ImageFolder(root/"train", transform=train_tfms)
    ds_va = datasets.ImageFolder(root/"valid", transform=eval_tfms)
    ds_te = datasets.ImageFolder(root/"test",  transform=eval_tfms)

    common = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, **common)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, **common)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, **common)
    return ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te

def build_vit_b16(num_classes: int):
    m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    in_f = m.heads.head.in_features
    m.heads.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, num_classes))
    return m

@torch.no_grad()
def evaluate(model, dl, loss_fn, device):
    model.eval()
    tot_loss = tot_corr = tot_cnt = 0
    for x,y in dl:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type!="cpu" else torch.bfloat16, enabled=(device.type!="cpu")):
            logits = model(x); loss = loss_fn(logits, y)
        tot_loss += loss.item()*x.size(0)
        tot_corr += (logits.argmax(1)==y).sum().item()
        tot_cnt  += x.size(0)
    return tot_loss/tot_cnt, tot_corr/tot_cnt

def train_one_epoch(model, dl, opt, loss_fn, scaler, device):
    model.train()
    tot_loss = tot_corr = tot_cnt = 0
    for x,y in dl:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type!="cpu" else torch.bfloat16, enabled=(device.type!="cpu")):
            logits = model(x); loss = loss_fn(logits, y)
        if scaler is not None and device.type=="cuda":
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        tot_loss += loss.item()*x.size(0)
        tot_corr += (logits.argmax(1)==y).sum().item()
        tot_cnt  += x.size(0)
    return tot_loss/tot_cnt, tot_corr/tot_cnt

class EarlyStopper:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience; self.min_delta = min_delta
        self.best = None; self.bad = 0
    def step(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss; self.bad = 0; return False
        self.bad += 1
        return self.bad >= self.patience

class WarmupCosine:
    def __init__(self, optimizer, base_lr, epochs, warmup_epochs):
        self.opt = optimizer
        self.base_lr = base_lr
        self.epochs = epochs
        self.warm = warmup_epochs
        self.t = 0
    def step(self):
        self.t += 1
        if self.t <= self.warm:
            lr = self.base_lr * self.t / max(1, self.warm)
        else:
            # cosine from base_lr to near 0 for remaining epochs
            progress = (self.t - self.warm) / max(1, self.epochs - self.warm)
            lr = 0.5 * self.base_lr * (1 + np.cos(np.pi * (1 - progress)))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr

def main():
    set_seed(SEED)
    device = pick_device()
    if device.type=="cuda":
        torch.set_float32_matmul_precision("high")
        print(f"[INFO] Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"[INFO] Device: {device}")

    data_root = (Path(__file__).resolve().parent / REL_DATA).resolve()
    if not (data_root/"train").exists(): raise SystemExit(f"[ERROR] {data_root}/train not found")

    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te = build_dataloaders(data_root)
    num_classes = len(ds_tr.classes)
    print(f"[INFO] Classes: {ds_tr.classes}")
    print(f"[INFO] Train/Valid/Test sizes: {len(ds_tr)}/{len(ds_va)}/{len(ds_te)}")

    model = build_vit_b16(num_classes).to(device)
    print(f"[INFO] Model params (trainable): {count_params(model)/1e6:.2f}M")

    for p in model.parameters(): p.requires_grad = False
    for p in model.heads.parameters(): p.requires_grad = True

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR/LOG_CSV
    init_csv_log(log_path)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda"))
    stopper = EarlyStopper()
    best_val_acc = 0.0
    best_path = OUT_DIR/"vit_b16_best.pt"

    sched = WarmupCosine(opt, base_lr=BASE_LR, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS)

    start_all = time.perf_counter()
    for epoch in range(1, EPOCHS+1):
        t0 = time.perf_counter()

        # разморозка после линейного пробинга
        if epoch == LINEAR_PROBE_EPOCHS + 1 and not all(p.requires_grad for p in model.parameters()):
            for p in model.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
            sched = WarmupCosine(opt, base_lr=BASE_LR, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS)
            print(f"[INFO] Unfroze backbone at epoch {epoch}")

        lr_now = sched.step()
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, opt, loss_fn, scaler, device)
        va_loss, va_acc = evaluate(model, dl_va, loss_fn, device)
        dt = time.perf_counter() - t0

        phase = "lp" if epoch <= LINEAR_PROBE_EPOCHS else "ft"
        print(f"[INFO] Epoch {epoch:02d}/{EPOCHS} [{phase}] | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {lr_now:.2e}")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, phase, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}", f"{lr_now:.6e}", f"{dt:.2f}"])

        if va_acc > best_val_acc + 1e-4:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict(), "classes": ds_tr.classes, "config": {"img_size": IMG_SIZE}}, best_path)
            print(f"[INFO] Saved best checkpoint: {best_path} (val_acc={best_val_acc:.4f})")

        if stopper.step(va_loss):
            print(f"[INFO] Early stopping at epoch {epoch} (no val_loss improvement for {PATIENCE} epochs).")
            break

    total_min = (time.perf_counter()-start_all)/60
    print(f"[INFO] Training finished in {total_min:.1f} min. Best val acc: {best_val_acc:.4f}")

    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    te_loss, te_acc = evaluate(model, dl_te, loss_fn, device)
    print(f"[INFO] Test loss {te_loss:.4f} | Test acc {te_acc:.4f}")

    correct = {i:0 for i in range(num_classes)}
    total   = {i:0 for i in range(num_classes)}
    model.eval()
    with torch.no_grad():
        for x,y in dl_te:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            for t,p in zip(y.tolist(), pred.tolist()):
                total[t]+=1; correct[t]+= int(p==t)
    pc = {ds_tr.classes[i]: (correct[i]/total[i] if total[i]>0 else 0.0) for i in range(num_classes)}
    with open(OUT_DIR/"test_per_class_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(pc, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved per-class accuracy and log CSV to: {OUT_DIR}")

if __name__ == "__main__":
    main()
