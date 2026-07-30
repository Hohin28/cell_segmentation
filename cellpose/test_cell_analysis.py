import os, sys, traceback

# Direct import of cell_analysis without going through cellpose package
sys.path.insert(0, r"C:\Users\Hohin.J\cellpose")

# Import only the needed module directly to avoid GUI/Qt import chain
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cell_analysis",
    r"C:\Users\Hohin.J\cellpose\cellpose\cell_analysis.py"
)
ca = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(ca)
    import_status = "IMPORT OK"
except Exception as e:
    import_status = f"IMPORT FAILED: {traceback.format_exc()}"

import numpy as np
masks = np.zeros((200, 200), dtype='uint16')
yy, xx = np.ogrid[:200, :200]
masks[(yy - 50)**2 + (xx - 50)**2 < 20**2] = 1
masks[(yy - 150)**2 + (xx - 150)**2 < 40**2] = 2

lines = [import_status]

try:
    n = ca.count_cells(masks)
    lines.append(f"COUNT: {n}")
except Exception as e:
    lines.append(f"COUNT ERROR: {traceback.format_exc()}")

try:
    bnd = ca.extract_boundaries(masks)
    lines.append(f"BOUNDARIES: {len(bnd)} cells")
    for b in bnd:
        lines.append(f"  cell {b['cell_id']}: area={b['area']} perim={b['perimeter']:.1f}")
except Exception as e:
    lines.append(f"BOUNDARY ERROR: {traceback.format_exc()}")

try:
    for mode in ['auto', 'blood', 'histology', 'papsmear']:
        cls = ca.classify_cells(masks, None, mode=mode)
        labels = {k: v['label'] for k, v in cls.items()}
        lines.append(f"CLASSIFY ({mode}): {labels}")
except Exception as e:
    lines.append(f"CLASSIFY ERROR: {traceback.format_exc()}")

try:
    sep = ca.separate_overlapping_cells(masks)
    lines.append(f"SEPARATION: {int(sep.max())} cells after")
except Exception as e:
    lines.append(f"SEPARATION ERROR: {traceback.format_exc()}")

lines.append("DONE")

with open("test_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
