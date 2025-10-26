from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = (PROJECT_ROOT / "data" / "modified").resolve()
OUT_DIR = Path(__file__).parent / "stats_out"
CSV_NAME_LONG = "class_distribution.csv"
CSV_NAME_WIDE = "class_distribution_wide.csv"

def list_images(p: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]

def count_split(split_dir: Path):
    records = []
    if not split_dir.exists():
        print(f"[WARN] Split not found: {split_dir}")
        return pd.DataFrame(columns=["split","class","count"])
    for cls_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
        c = len(list_images(cls_dir))
        records.append({"split": split_dir.name, "class": cls_dir.name, "count": c})
    df = pd.DataFrame(records)
    if df.empty:
        print(f"[WARN] No classes found in {split_dir}")
    return df

def plot_split(df: pd.DataFrame, out_path: Path, title: str):
    if df.empty:
        print(f"[WARN] Empty data for {title}, skipping plot")
        return
    df = df.sort_values("class")
    plt.figure(figsize=(10, 6))
    plt.bar(df["class"], df["count"])
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved plot: {out_path}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dfs = []
    for split in ["train", "valid", "test"]:
        split_dir = DATA_ROOT / split
        df = count_split(split_dir)
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True).fillna(0)
    if all_df.empty:
        print("[ERROR] No data found. Check your dataset structure.")
        return
    csv_path = OUT_DIR / CSV_NAME_LONG
    all_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved CSV: {csv_path}")
    for split in ["train", "valid", "test"]:
        df_split = all_df[all_df["split"] == split]
        plot_split(df_split, OUT_DIR / f"class_distribution_{split}.png", f"Class Distribution - {split}")
    wide = all_df.pivot_table(index="class", columns="split", values="count", fill_value=0).reset_index()
    wide_path = OUT_DIR / CSV_NAME_WIDE
    wide.to_csv(wide_path, index=False)
    print(f"[INFO] Saved wide CSV: {wide_path}")

if __name__ == "__main__":
    main()
