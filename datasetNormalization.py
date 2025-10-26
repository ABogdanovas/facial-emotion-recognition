from pathlib import Path
import json
from collections import defaultdict
from PIL import Image, ImageOps

ROOT = Path("C:/git/facial-emotion-recognition/data/default")
OUT  = Path("C:/git/facial-emotion-recognition/data/modified")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 224
MARGIN_RATIO = 0.08
MIN_BOX = 24

EXCLUDE_CLASSES = {"content"}

def find_coco_json(split_dir: Path) -> Path:
    cand = list(split_dir.glob("*_annotations.coco.json"))
    if cand:
        return cand[0]
    cand = list((split_dir / "annotations").glob("*.json"))
    if cand:
        return cand[0]
    raise FileNotFoundError(f"Cannot find COCO JSON in {split_dir}")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def to_square_squash(img: Image.Image, tgt: int) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    return img.resize((tgt, tgt), Image.BICUBIC)

def process_split(split_name: str):
    split_dir = ROOT / split_name
    ann_path = find_coco_json(split_dir)
    images_dir = split_dir / "images"
    if not images_dir.exists():
        images_dir = split_dir

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in coco["images"]}
    id2cat  = {c["id"]: c["name"] for c in coco["categories"] if c["name"].lower() != "emotions"}

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        if "bbox" in a and a.get("iscrowd", 0) == 0 and a.get("category_id") in id2cat:
            x, y, w, h = a["bbox"]
            if w >= MIN_BOX and h >= MIN_BOX:
                anns_by_img[a["image_id"]].append(a)

    out_split = OUT / split_name
    out_split.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_small = 0
    missing_files = 0
    excluded = 0

    for img_id, anns in anns_by_img.items():
        img_path = images_dir / id2file[img_id]
        if not img_path.exists():
            missing_files += 1
            continue

        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        W, H = im.size

        for i, a in enumerate(anns):
            cls_name = id2cat[a["category_id"]].lower()
            if cls_name in EXCLUDE_CLASSES:
                excluded += 1
                continue

            x, y, w, h = a["bbox"]
            dx = int(round(w * MARGIN_RATIO))
            dy = int(round(h * MARGIN_RATIO))
            x0 = clamp(int(x) - dx, 0, W - 1)
            y0 = clamp(int(y) - dy, 0, H - 1)
            x1 = clamp(int(x + w) + dx, 0, W)
            y1 = clamp(int(y + h) + dy, 0, H)

            cw, ch = x1 - x0, y1 - y0
            if cw < MIN_BOX or ch < MIN_BOX:
                skipped_small += 1
                continue

            crop = im.crop((x0, y0, x1, y1))
            crop = to_square_squash(crop, TARGET_SIZE)

            out_dir = out_split / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(id2file[img_id]).stem
            out_file = out_dir / f"{stem}_{i}.jpg"
            crop.save(out_file, quality=95)
            saved += 1

    print(f"[{split_name}] saved: {saved}, excluded(content): {excluded}, skipped_small: {skipped_small}, missing_files: {missing_files}")

if __name__ == "__main__":
    for sp in ["train", "valid", "test"]:
        process_split(sp)
    print("DONE.")
