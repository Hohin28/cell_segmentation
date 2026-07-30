"""
=============================================================================
SIPAKMED PREPARATION SCRIPT
=============================================================================
Copies only the .bmp cell images from the SIPaKMeD CROPPED folders into the
4-class training_crops/ directory structure required by the classifier.

Mapping:
  im_Dyskeratotic/CROPPED/*.bmp   -> training_crops/HSIL/
  im_Koilocytotic/CROPPED/*.bmp   -> training_crops/LSIL/
  im_Metaplastic/CROPPED/*.bmp    -> training_crops/ASCUS/
  im_Parabasal/CROPPED/*.bmp      -> training_crops/NILM/
  im_Superficial-Intermediate/CROPPED/*.bmp -> training_crops/NILM/

IMPORTANT:
  - Only .bmp image files are copied (NOT .dat annotation files).
  - Filenames are prefixed with the source class to prevent overwrite collisions
    when merging Parabasal + Superficial-Intermediate into a single NILM folder.
  - This script does NOT touch Cellpose-SAM or the GUI.
=============================================================================
"""

import os
import shutil
from pathlib import Path

SIPAKMED_ROOT = r"dataset_downloads\SIPaKMeD"
OUTPUT_ROOT   = r"training_crops"

CLASS_MAP = {
    "im_Dyskeratotic"           : "HSIL",
    "im_Koilocytotic"           : "LSIL",
    "im_Metaplastic"            : "ASCUS",
    "im_Parabasal"              : "NILM",
    "im_Superficial-Intermediate": "NILM",
}

def find_cropped_folder(class_name):
    p = Path(SIPAKMED_ROOT) / class_name / class_name / "CROPPED"
    return p if p.is_dir() else None

def prepare():
    print("=" * 65)
    print("  SIPAKMED DATASET PREPARATION")
    print("=" * 65)

    if not Path(SIPAKMED_ROOT).is_dir():
        print(f"\n[FATAL] SIPaKMeD root not found: {os.path.abspath(SIPAKMED_ROOT)}")
        return

    # Create all output class dirs
    for cls in set(CLASS_MAP.values()):
        out_dir = Path(OUTPUT_ROOT) / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {out_dir}")

    print()
    total_copied  = 0
    total_skipped = 0
    per_class     = {}

    for src_class, bethesda_cls in CLASS_MAP.items():
        folder = find_cropped_folder(src_class)
        if folder is None:
            print(f"  [SKIP] CROPPED folder not found for: {src_class}")
            continue

        out_dir = Path(OUTPUT_ROOT) / bethesda_cls
        copied  = 0
        skipped = 0

        for bmp_file in sorted(folder.glob("*.bmp")):
            # Prefix filename with source class name to avoid NILM name collisions
            dest_name = f"{src_class}_{bmp_file.name}"
            dest_path = out_dir / dest_name

            if dest_path.exists():
                skipped += 1
            else:
                shutil.copy2(bmp_file, dest_path)
                copied += 1

        total_copied  += copied
        total_skipped += skipped
        per_class[bethesda_cls] = per_class.get(bethesda_cls, 0) + copied

        print(f"  {src_class:<38} -> {bethesda_cls:<6}  "
              f"copied: {copied:>4}  skipped: {skipped:>4}")

    # Final summary
    print("\n" + "=" * 65)
    print("  FINAL CLASS COUNTS IN training_crops/")
    print("=" * 65)
    grand_total = 0
    for cls in ["HSIL", "LSIL", "ASCUS", "NILM"]:
        n = sum(
            1 for f in (Path(OUTPUT_ROOT) / cls).glob("*.bmp")
        ) if (Path(OUTPUT_ROOT) / cls).is_dir() else 0
        grand_total += n
        print(f"    {cls:<10}: {n} images")

    print(f"\n    TOTAL   : {grand_total} images")
    print(f"    Copied  : {total_copied}")
    print(f"    Skipped : {total_skipped} (already existed)")
    print("\n  ✅ Preparation complete.")
    print("     Next step: python cervical_classifier_train.py")
    print("=" * 65)

if __name__ == "__main__":
    prepare()
