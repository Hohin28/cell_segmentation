"""
=============================================================================
SIPAKMED DATASET ANALYSIS SCRIPT
=============================================================================
Analyzes the SIPaKMeD dataset directly from the CROPPED sub-folders.

Folder structure expected:
  dataset_downloads/SIPaKMeD/
    im_Dyskeratotic/im_Dyskeratotic/CROPPED/     -> HSIL
    im_Koilocytotic/im_Koilocytotic/CROPPED/     -> LSIL
    im_Metaplastic/im_Metaplastic/CROPPED/       -> ASCUS
    im_Parabasal/im_Parabasal/CROPPED/           -> NILM
    im_Superficial-Intermediate/im_Superficial-Intermediate/CROPPED/ -> NILM

Reports:
  1. Number of .bmp cell images per source class
  2. Bethesda class mapping
  3. Final 4-class distribution after merging Parabasal + Superficial -> NILM
  4. Imbalance ratio
  5. Min image dimension per class
  6. Go/No-Go training decision
=============================================================================
"""

import os
from pathlib import Path

# ─── Dataset Root ──────────────────────────────────────────────────────────────
SIPAKMED_ROOT = r"dataset_downloads\SIPaKMeD"

# ─── Source → Bethesda Mapping ────────────────────────────────────────────────
CLASS_MAP = {
    "im_Dyskeratotic":           "HSIL",
    "im_Koilocytotic":           "LSIL",
    "im_Metaplastic":            "ASCUS",
    "im_Parabasal":              "NILM",
    "im_Superficial-Intermediate": "NILM",
}

# SIPaKMeD nests each class inside a same-named subfolder, then CROPPED/
# e.g. im_Dyskeratotic/im_Dyskeratotic/CROPPED/

def find_cropped_folder(class_folder_name):
    """Resolve the CROPPED/ path inside each class directory."""
    p = Path(SIPAKMED_ROOT) / class_folder_name / class_folder_name / "CROPPED"
    return p if p.is_dir() else None

def count_bmp(folder: Path) -> int:
    return sum(1 for f in folder.iterdir() if f.suffix.lower() == ".bmp")

def get_dimensions(folder: Path, sample_n=5):
    """Try to read up to sample_n images and return min width/height."""
    try:
        from PIL import Image
        dims = []
        for f in list(folder.glob("*.bmp"))[:sample_n]:
            with Image.open(f) as img:
                dims.append(img.size)  # (W, H)
        if dims:
            min_w = min(d[0] for d in dims)
            min_h = min(d[1] for d in dims)
            return f"{min_w}x{min_h}"
    except Exception:
        pass
    return "N/A (Pillow not installed)"

# ─── Main Analysis ────────────────────────────────────────────────────────────
print("=" * 65)
print("  SIPAKMED DATASET ANALYSIS REPORT")
print("=" * 65)

if not Path(SIPAKMED_ROOT).is_dir():
    print(f"\n[FATAL] SIPaKMeD root not found: {os.path.abspath(SIPAKMED_ROOT)}")
    print("        Run setup_dataset.py first.")
    exit(1)

source_counts = {}
bethesda_counts = {}
min_dims = {}

print(f"\n{'Source Class':<35} {'BMP Images':>11}  {'→ Bethesda Class'}")
print("-" * 65)

for src_class, bethesda_cls in CLASS_MAP.items():
    folder = find_cropped_folder(src_class)
    if folder is None:
        print(f"  {src_class:<33} {'MISSING':>11}  → {bethesda_cls} ❌")
        source_counts[src_class] = 0
        continue

    n = count_bmp(folder)
    source_counts[src_class] = n
    bethesda_counts[bethesda_cls] = bethesda_counts.get(bethesda_cls, 0) + n

    dim_str = get_dimensions(folder)
    print(f"  {src_class:<33} {n:>11}  → {bethesda_cls}  (sample dims: {dim_str})")

# ─── Summary ──────────────────────────────────────────────────────────────────
total_images = sum(source_counts.values())

print("\n" + "=" * 65)
print("  FINAL 4-CLASS DISTRIBUTION (after NILM merge)")
print("=" * 65)

MINIMUM_GOOD = 300
MINIMUM_WARNING = 100
MINIMUM_CRITICAL = 50

class_order = ["HSIL", "LSIL", "ASCUS", "NILM"]
print(f"\n{'Bethesda Class':<12} {'Images':>8}   {'Status'}")
print("-" * 50)

has_critical = False
for cls in class_order:
    n = bethesda_counts.get(cls, 0)
    if n == 0:
        status = "❌  EMPTY"
        has_critical = True
    elif n < MINIMUM_CRITICAL:
        status = f"🔴  CRITICAL (<{MINIMUM_CRITICAL})"
        has_critical = True
    elif n < MINIMUM_WARNING:
        status = f"🟠  WARNING (<{MINIMUM_WARNING})"
    elif n < MINIMUM_GOOD:
        status = f"🟡  ACCEPTABLE (<{MINIMUM_GOOD})"
    else:
        status = "🟢  GOOD"
    print(f"  {cls:<12} {n:>8}   {status}")

print(f"\n  TOTAL IMAGES : {total_images}")

# ─── Imbalance ────────────────────────────────────────────────────────────────
valid = [v for v in bethesda_counts.values() if v > 0]
if len(valid) >= 2:
    max_n = max(valid)
    min_n = min(valid)
    ratio = max_n / min_n
    max_cls = [k for k, v in bethesda_counts.items() if v == max_n][0]
    min_cls = [k for k, v in bethesda_counts.items() if v == min_n][0]
    print(f"\n  Largest class  : {max_cls} ({max_n} images)")
    print(f"  Smallest class : {min_cls} ({min_n} images)")
    print(f"  Imbalance Ratio: {ratio:.1f}x")
    if ratio > 5:
        print("  ⚠️  Class-Weighted Loss will be applied during training.")
    else:
        print("  ✅  Imbalance is within acceptable range.")

# ─── Expected splits ─────────────────────────────────────────────────────────
print("\n  Expected Train / Val / Test Splits (70 / 15 / 15):")
print(f"    Train : {int(total_images * 0.70)}")
print(f"    Val   : {int(total_images * 0.15)}")
print(f"    Test  : {total_images - int(total_images * 0.70) - int(total_images * 0.15)}")

# ─── Training path ───────────────────────────────────────────────────────────
print("\n  OUTPUT FOLDER THAT WILL BE CREATED:")
for cls in class_order:
    print(f"    training_crops\\{cls}\\")

print("\n" + "=" * 65)
if has_critical:
    print("  ❌ DATASET NOT READY — Critical class(es) need data.")
else:
    print("  ✅ DATASET READY — Run prepare_sipakmed.py to copy into training_crops/")
    print("     Then run: python cervical_classifier_train.py")
print("=" * 65)
