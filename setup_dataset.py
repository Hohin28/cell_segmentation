"""
=============================================================================
CERVICAL DATASET SETUP SCRIPT
=============================================================================
This script manages the complete dataset preparation pipeline:

1. Checks for Kaggle API credentials and guides setup if missing.
2. Downloads SIPaKMeD and Herlev from Kaggle (both are FREE, publicly licensed).
3. Automatically organizes images into the 8 Bethesda class folders.
4. Generates a full report on class distribution and download status.
5. Does NOT start training — training must be triggered manually via
   cervical_classifier_train.py after reviewing this report.

LICENSING:
  SIPaKMeD   — CC BY-NC-SA 4.0 (Free for non-commercial research)
               Cite: Plissiti et al., ICIP 2018.
  Herlev     — Free for non-commercial academic research.
               Cite: Jantzen et al., 2005.

DATASET → BETHESDA CLASS MAPPING:
  SIPaKMeD:
    im_Dyskeratotic      → HSIL
    im_Koilocytotic      → LSIL
    im_Metaplastic       → ASCUS
    im_Parabasal         → NILM
    im_Superficial       → NILM

  Herlev:
    severe_dysplastic    → HSIL
    high_grade_dysplastic→ ASC-H
    moderate_dysplastic  → LSIL
    mild_dysplastic      → LSIL
    normal_columnar      → ENDO
    normal_intermediate  → NILM
    normal_superficial   → NILM

NOTE: SCC, INFL classes are not covered by these two datasets.
After setup you will be told exactly which classes still need more data.
=============================================================================
"""

import os
import sys
import shutil
import json
import zipfile
from pathlib import Path
from collections import defaultdict

# ─── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR      = "training_crops"
DOWNLOAD_DIR    = "dataset_downloads"
KAGGLE_DIR      = os.path.join(os.path.expanduser("~"), ".kaggle")
KAGGLE_JSON     = os.path.join(KAGGLE_DIR, "kaggle.json")

REQUIRED_CLASSES = ["ASC-H", "ASCUS", "ENDO", "HSIL", "INFL", "LSIL", "NILM", "SCC"]
IMG_EXTENSIONS   = ('.png', '.jpg', '.jpeg', '.bmp')

# ─── Dataset Definitions ──────────────────────────────────────────────────────
DATASETS = {
    "SIPaKMeD": {
        "kaggle_id"  : "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed",
        "license"    : "CC BY-NC-SA 4.0 — Free for non-commercial research.",
        "citation"   : "Plissiti et al., SIPAKMED, ICIP 2018.",
        "classes"    : {
            "im_Dyskeratotic"      : "HSIL",
            "im_Koilocytotic"      : "LSIL",
            "im_Metaplastic"       : "ASCUS",
            "im_Parabasal"         : "NILM",
            "im_Superficial-Intermediate": "NILM",
        },
        "covers"     : ["HSIL", "LSIL", "ASCUS", "NILM"],
        "does_not_cover": ["SCC", "ASC-H", "ENDO", "INFL"],
    },
    "Herlev": {
        "kaggle_id"  : "yuvrajsingh/herlev-dataset",
        "license"    : "Free for non-commercial academic research.",
        "citation"   : "Jantzen et al., Pap-smear benchmark data, 2005.",
        "classes"    : {
            "severe_dysplastic"    : "HSIL",
            "high_grade_dysplastic": "ASC-H",
            "moderate_dysplastic"  : "LSIL",
            "mild_dysplastic"      : "LSIL",
            "normal_columnar"      : "ENDO",
            "normal_intermediate"  : "NILM",
            "normal_superficial"   : "NILM",
        },
        "covers"     : ["HSIL", "ASC-H", "LSIL", "ENDO", "NILM"],
        "does_not_cover": ["SCC", "ASCUS", "INFL"],
    }
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def _count_images(folder):
    if not os.path.isdir(folder): return 0
    return sum(1 for f in Path(folder).rglob("*") if f.suffix.lower() in IMG_EXTENSIONS)

# ─── Step 1: Kaggle Credential Check ─────────────────────────────────────────
def check_kaggle_credentials():
    _print_header("STEP 1: KAGGLE API CREDENTIAL CHECK")

    if os.path.exists(KAGGLE_JSON):
        try:
            with open(KAGGLE_JSON) as f:
                creds = json.load(f)
            if "username" in creds and "key" in creds:
                print(f"  ✅ kaggle.json found: {KAGGLE_JSON}")
                print(f"     Username : {creds['username']}")
                print(f"     Key      : {creds['key'][:6]}...")
                return True
        except Exception as e:
            print(f"  ❌ kaggle.json is malformed: {e}")

    print("""
  ❌ Kaggle API credentials NOT found.

  To enable automatic downloading, follow these steps:

  1. Go to https://www.kaggle.com and log in.
  2. Click your profile picture → Account → scroll to API section.
  3. Click "Create New API Token". A file called 'kaggle.json' will download.
  4. Place that file here:
       C:\\Users\\Hohin.J\\.kaggle\\kaggle.json

  5. Re-run this script.

  IF YOU CANNOT DO THIS NOW — skip to the manual download instructions
  printed at the end of this script.
""")
    return False


# ─── Step 2: Download Datasets ────────────────────────────────────────────────
def download_datasets():
    _print_header("STEP 2: DOWNLOADING DATASETS (Kaggle API)")

    try:
        import kaggle
    except ImportError:
        print("  Installing kaggle Python library...")
        os.system(f"{sys.executable} -m pip install kaggle -q")
        import kaggle

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    results = {}

    for name, info in DATASETS.items():
        dest = os.path.join(DOWNLOAD_DIR, name)
        if os.path.isdir(dest) and _count_images(dest) > 0:
            print(f"\n  [SKIP] {name} already downloaded ({_count_images(dest)} images found).")
            results[name] = dest
            continue

        print(f"\n  Downloading {name} ...")
        print(f"    Kaggle ID : {info['kaggle_id']}")
        print(f"    License   : {info['license']}")
        os.makedirs(dest, exist_ok=True)
        try:
            os.system(
                f"kaggle datasets download -d {info['kaggle_id']} "
                f"--path {dest} --unzip -q"
            )
            count = _count_images(dest)
            print(f"    ✅ Download complete — {count} images found.")
            results[name] = dest
        except Exception as e:
            print(f"    ❌ Download failed: {e}")
            results[name] = None

    return results


# ─── Step 3: Organize into Bethesda Folders ──────────────────────────────────
def organize_into_classes(download_results):
    _print_header("STEP 3: ORGANIZING INTO BETHESDA CLASS FOLDERS")

    for cls in REQUIRED_CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    moved_counts = defaultdict(int)
    skipped      = 0

    for dataset_name, dataset_path in download_results.items():
        if dataset_path is None:
            print(f"  [SKIP] {dataset_name} was not downloaded successfully.")
            continue

        class_map = DATASETS[dataset_name]["classes"]
        print(f"\n  Processing {dataset_name} from: {dataset_path}")

        for root, dirs, files in os.walk(dataset_path):
            folder_name = os.path.basename(root)

            # Find which Bethesda class this folder maps to
            bethesda_cls = None
            for src_key, target_cls in class_map.items():
                if src_key.lower() in folder_name.lower():
                    bethesda_cls = target_cls
                    break

            if bethesda_cls is None:
                continue

            # Copy images into the target folder
            for fname in files:
                if fname.lower().endswith(IMG_EXTENSIONS):
                    src_path  = os.path.join(root, fname)
                    # Build unique name: datasetName_originalFolder_filename.ext
                    safe_name = f"{dataset_name}_{folder_name}_{fname}"
                    dst_path  = os.path.join(OUTPUT_DIR, bethesda_cls, safe_name)
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                        moved_counts[bethesda_cls] += 1
                    else:
                        skipped += 1

    print(f"\n  Files copied per class:")
    for cls in REQUIRED_CLASSES:
        count = moved_counts.get(cls, 0)
        print(f"    {cls:<10}: {count:>5} images copied")
    if skipped:
        print(f"\n  {skipped} files skipped (already existed in destination).")

    return moved_counts


# ─── Step 4: Generate Full Report ────────────────────────────────────────────
def generate_report(download_results, moved_counts):
    _print_header("STEP 4: DATASET PREPARATION REPORT")

    # Dataset sources
    print("\n  ── Dataset Sources & Licensing ──────────────────────────────")
    for name, info in DATASETS.items():
        status = "✅ Downloaded" if download_results.get(name) else "❌ Not Downloaded"
        print(f"\n  {name} [{status}]")
        print(f"    License  : {info['license']}")
        print(f"    Citation : {info['citation']}")
        print(f"    Covers   : {info['covers']}")
        print(f"    Missing  : {info['does_not_cover']}")

    # Final class counts in training_crops/
    print("\n  ── Final Class Distribution in training_crops/ ──────────────")
    total   = 0
    valid   = True

    MINIMUM_IMAGES_CRITICAL = 50
    MINIMUM_IMAGES_WARNING  = 100
    MINIMUM_IMAGES_GOOD     = 300

    counts = {}
    for cls in REQUIRED_CLASSES:
        d = os.path.join(OUTPUT_DIR, cls)
        n = _count_images(d)
        counts[cls] = n
        total += n

        if n == 0:
            status = "❌  EMPTY"
            valid  = False
        elif n < MINIMUM_IMAGES_CRITICAL:
            status = "🔴  CRITICAL (<50)"
            valid  = False
        elif n < MINIMUM_IMAGES_WARNING:
            status = "🟠  WARNING (<100)"
        elif n < MINIMUM_IMAGES_GOOD:
            status = "🟡  ACCEPTABLE (<300)"
        else:
            status = "🟢  GOOD"
        print(f"    {cls:<10}: {n:>5} images   {status}")

    print(f"\n    TOTAL: {total} images across {len(REQUIRED_CLASSES)} classes")

    # Classes still missing data
    missing_data = [c for c in REQUIRED_CLASSES if counts[c] < MINIMUM_IMAGES_CRITICAL]
    if missing_data:
        print(f"\n  ⚠️  Classes requiring additional data: {missing_data}")
        print("""
  RECOMMENDED ACTIONS:

  SCC class:
    The SCC class is not available in SIPaKMeD or Herlev.
    You have SCC slides in your Google Drive datasets (CCHRC, Bialystok).
    Manually copy SCC cell crops from those datasets to:
      training_crops\\SCC\\

  INFL class:
    Inflammatory cells are not labelled separately in SIPaKMeD or Herlev.
    Source from CCHRC dataset or use a general inflammatory cell dataset.
    Copy to: training_crops\\INFL\\
""")

    print("\n  ── Manual Download Instructions (If Kaggle Download Failed) ────")
    print("""
  SIPaKMeD:
    URL   : https://www.cs.uoi.gr/~marina/sipakmed.html
    OR    : https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed
    Action: Download the ZIP, extract it, and run this script again.
            The script will organize it automatically from dataset_downloads/SIPaKMeD/

  Herlev:
    URL   : https://www.kaggle.com/datasets/yuvrajsingh/herlev-dataset
    Action: Same as above — extract to dataset_downloads/Herlev/

  Your Private Datasets (CCHRC, Bialystok, SIPaKMeD 2025):
    Action: Download from your Google Drive and extract to:
              dataset_downloads/CCHRC/
              dataset_downloads/Bialystok/
            Then re-run this script and it will organize them automatically.
""")

    print("\n" + "=" * 65)
    if valid:
        print("  ✅ DATASET READY — Run cervical_classifier_train.py to begin training.")
    else:
        print("  ❌ DATASET NOT READY — Populate the empty/critical classes first.")
    print("=" * 65)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _print_header("CERVICAL DATASET AUTOMATED SETUP")
    print("  This script downloads, organizes, and validates your training dataset.")
    print("  Training will NOT start from this script. Training is in:")
    print("    cervical_classifier_train.py")

    has_kaggle = check_kaggle_credentials()

    if has_kaggle:
        download_results = download_datasets()
    else:
        print("\n  Attempting to organize any manually downloaded data...")
        download_results = {
            name: os.path.join(DOWNLOAD_DIR, name)
            if os.path.isdir(os.path.join(DOWNLOAD_DIR, name)) else None
            for name in DATASETS
        }

    moved_counts = organize_into_classes(download_results)
    generate_report(download_results, moved_counts)
