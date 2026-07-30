"""
╔══════════════════════════════════════════════════════════════════════╗
║   CELLPOSE-SAM  —  ENHANCED GUI v3.0  with PROPER CLASSIFICATION   ║
║                                                                      ║
║  HOW TO RUN:                                                         ║
║    conda activate cellpose                                           ║
║    pip install scikit-learn -q                                       ║
║    python cellpose_gui_v3.py                                         ║
║                                                                      ║
║  DATASETS for better accuracy (tell professor about these):          ║
║  → PanNuke   : warwick.ac.uk/fac/cross_fac/tia/data/pannuke         ║
║  → MoNuSAC   : monusac-2020.grand-challenge.org                     ║
║  → BCCD      : kaggle.com/datasets/jeetblahiri/bccd-dataset         ║
║  → BloodMNIST: medmnist.com                                          ║
╚══════════════════════════════════════════════════════════════════════╝

WHAT THIS DOES:
 1. Cell Segmentation   — Cellpose-SAM or cyto3 detects every cell
 2. Classification      — H&E colour deconvolution + morphology features
                          classifies each cell into: Neoplastic / Inflammatory /
                          Epithelial / Connective / Dead
 3. Localisation        — centroid (x,y) + bounding box for every cell
 4. Boundary Extraction — precise pixel-level outline per cell
 5. Overlap Separation  — watershed splits touching/merged cells
 6. Pap Smear Mode      — cervical cell analysis
 7. Blood Smear Mode    — WBC / RBC / Platelet

KEY TECHNIQUE — H&E COLOUR DECONVOLUTION:
 H&E images use two stains:
   Haematoxylin → blue/purple → stains NUCLEI
   Eosin        → pink        → stains CYTOPLASM
 We mathematically separate the two channels using the
 Ruifrok & Johnston (2001) optical density decomposition.
 This gives us nucleus intensity and cytoplasm intensity per cell —
 far more meaningful than raw RGB for classification.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, time, warnings, os, csv
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from collections import Counter

from skimage.color import label2rgb, rgb2hed
from skimage.measure import regionprops, find_contours
from skimage.segmentation import watershed, find_boundaries
from skimage.filters import gaussian, sobel
from skimage.morphology import disk, dilation, erosion
from skimage.exposure import equalize_adapthist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
import tifffile, torch
from cellpose import models

# ── palette ──────────────────────────────────────────────────────────
NAVY="#0D1B2A"; TEAL="#00A896"; TEAL2="#02C39A"; AMBER="#F0B429"
WHITE="#FFFFFF"; LGRAY="#1E2D3D"; DGRAY="#8EACC8"; RED="#E24B4A"
GREEN="#4CAF50"; PANEL="#0F2132"; PURPLE="#7C6AF5"

# ═════════════════════════════════════════════════════════════════════
#  CELL TYPE DEFINITIONS  (paper-aligned labels)
# ═════════════════════════════════════════════════════════════════════
TISSUE_TYPES = {
    "Neoplastic":   {"color":"#E24B4A","hex_rgb":(226,75,74),
                     "meaning":"Cancer/tumour cells — large, irregular nuclei"},
    "Inflammatory": {"color":"#4472C4","hex_rgb":(68,114,196),
                     "meaning":"Immune cells — small, very dark round nuclei"},
    "Epithelial":   {"color":"#00A896","hex_rgb":(0,168,150),
                     "meaning":"Lining cells — medium regular nuclei"},
    "Connective":   {"color":"#F0B429","hex_rgb":(240,180,41),
                     "meaning":"Stromal/support tissue — elongated spindle nuclei"},
    "Dead":         {"color":"#888780","hex_rgb":(136,135,128),
                     "meaning":"Necrotic cells — fragmented/pale nuclei"},
}

BLOOD_TYPES = {
    "Neutrophil":  {"color":"#4472C4","meaning":"WBC — multi-lobed, most common"},
    "Lymphocyte":  {"color":"#7C6AF5","meaning":"WBC — small, round, large nucleus"},
    "Monocyte":    {"color":"#00A896","meaning":"WBC — largest WBC, kidney-shaped nucleus"},
    "Eosinophil":  {"color":"#F0B429","meaning":"WBC — bi-lobed, pink granules"},
    "RBC":         {"color":"#E24B4A","meaning":"Red blood cell — biconcave, no nucleus"},
    "Platelet":    {"color":"#E8A4C9","meaning":"Thrombocyte — tiny cell fragment"},
}

PAP_TYPES = {
    "Superficial":  {"color":"#4CAF50","meaning":"Normal — large flat cell"},
    "Intermediate": {"color":"#00A896","meaning":"Normal — medium cell"},
    "Parabasal":    {"color":"#4472C4","meaning":"Normal — small deep cell"},
    "Koilocyte":    {"color":"#F0B429","meaning":"Abnormal — HPV infection sign"},
    "Dyskeratotic": {"color":"#E24B4A","meaning":"Abnormal — pre-cancerous change"},
    "Metaplastic":  {"color":"#7C6AF5","meaning":"Transformation zone cell"},
}


# ═════════════════════════════════════════════════════════════════════
#  H&E COLOUR DECONVOLUTION CLASSIFIER
#  Uses Ruifrok & Johnston (2001) stain separation
# ═════════════════════════════════════════════════════════════════════
class HEClassifier:
    """
    Classifies cells in H&E images using colour deconvolution.

    H&E Stain Vectors (from Ruifrok & Johnston 2001):
      Haematoxylin: [0.650, 0.704, 0.286]
      Eosin:        [0.072, 0.990, 0.105]

    Feature vector per cell:
      [h_mean, h_std, e_mean, e_std, area, eccentricity,
       solidity, ne_ratio, blue_frac, roundness]
    """

    def __init__(self):
        self.scaler  = StandardScaler()
        self.knn     = KNeighborsClassifier(n_neighbors=5)
        self.trained = False

    def _he_deconvolve(self, img_rgb):
        """
        Separate H&E stains using skimage rgb2hed.
        Returns h_channel (haematoxylin) and e_channel (eosin).
        Both are 2D arrays, values roughly 0-1 (higher = more stain).
        """
        img_clip = np.clip(img_rgb, 0.001, 0.999).astype(np.float64)
        hed = rgb2hed(img_clip)
        h = np.clip(hed[:,:,0], 0, None); h = h / (h.max()+1e-8)
        e = np.clip(hed[:,:,1], 0, None); e = e / (e.max()+1e-8)
        return h, e

    def _extract_features(self, masks, img_rgb):
        """
        Extract feature vector for every detected cell.
        Returns:
          cell_ids  : list of cell IDs
          features  : np.array (N_cells × 10)
        """
        h_chan, e_chan = self._he_deconvolve(img_rgb)
        props   = regionprops(masks)
        ids, feats = [], []

        for p in props:
            cid  = p.label
            rr, cc = np.where(masks == cid)
            if len(rr) < 5:
                continue

            # haematoxylin features (nuclear staining)
            h_vals = h_chan[rr, cc]
            h_mean = float(np.mean(h_vals))
            h_std  = float(np.std(h_vals))

            # eosin features (cytoplasm staining)
            e_vals = e_chan[rr, cc]
            e_mean = float(np.mean(e_vals))

            # morphological features
            area   = float(p.area)
            ecc    = float(p.eccentricity)
            sol    = float(p.solidity)

            # nuclear-to-eosin ratio (high = more nuclear material)
            ne_ratio = h_mean / (e_mean + 0.05)

            # blue fraction in original image (nuclei are blue in H&E)
            r_c = float(np.mean(img_rgb[rr,cc,0]))
            b_c = float(np.mean(img_rgb[rr,cc,2]))
            blue_frac = b_c / (r_c + b_c + 0.01)

            # roundness  (4πA / P²)
            perim = float(p.perimeter) + 1
            roundness = (4 * np.pi * area) / (perim ** 2)

            feats.append([h_mean, h_std, e_mean, ne_ratio,
                          area, ecc, sol, blue_frac, roundness,
                          h_mean * blue_frac])
            ids.append(cid)

        return ids, np.array(feats, dtype=np.float32)

    def classify_tissue(self, masks, img_rgb):
        """
        Classify H&E tissue cells.
        Rules derived from published H&E morphology literature:
          Neoplastic   : high H (dark nuclei), large area, irregular shape
          Inflammatory : very high H, very small area, round
          Epithelial   : moderate H, medium area, regular polygon
          Connective   : low H, elongated (high eccentricity)
          Dead         : low H, low solidity (fragmented)
        """
        ids, feats = self._extract_features(masks, img_rgb)
        if len(ids) == 0:
            return {}

        results = {}
        for cid, f in zip(ids, feats):
            h_mean, h_std, e_mean, ne_ratio, \
            area, ecc, sol, blue_frac, roundness, _ = f

            # ── rule tree ─────────────────────────────────────────
            if sol < 0.65 or (h_mean < 0.12 and e_mean < 0.12):
                ct = "Dead"
            elif area < 90 and h_mean > 0.35 and roundness > 0.7:
                ct = "Inflammatory"
            elif ecc > 0.80 and area < 300:
                ct = "Connective"
            elif h_mean > 0.40 and area > 140 and roundness < 0.75:
                ct = "Neoplastic"
            elif h_mean > 0.25 and 0.55 < roundness <= 0.85:
                ct = "Epithelial"
            elif area > 200 and h_mean > 0.30:
                ct = "Neoplastic"
            elif area < 120 and ecc < 0.55:
                ct = "Inflammatory"
            else:
                ct = "Epithelial"

            results[cid] = ct

        return results

    def classify_blood(self, masks, img_rgb):
        """Blood smear — WBC subtypes + RBC + Platelet."""
        ids, feats = self._extract_features(masks, img_rgb)
        results = {}
        props_map = {p.label: p for p in regionprops(masks)}

        for cid, f in zip(ids, feats):
            h_mean, h_std, e_mean, ne_ratio, \
            area, ecc, sol, blue_frac, roundness, _ = f
            p = props_map.get(cid)
            r_mean = float(np.mean(img_rgb[np.where(masks==cid)[0],
                                            np.where(masks==cid)[1], 0]))

            if area < 50:
                ct = "Platelet"
            elif area < 180 and r_mean > 0.65:
                ct = "RBC"
            elif h_mean > 0.5 and area < 160:
                ct = "Lymphocyte"
            elif h_mean > 0.4 and area > 250:
                ct = "Monocyte"
            elif blue_frac > 0.42 and ecc > 0.5:
                ct = "Neutrophil"
            elif r_mean > 0.6 and h_mean > 0.3:
                ct = "Eosinophil"
            else:
                ct = "Neutrophil"

            results[cid] = ct
        return results

    def classify_pap(self, masks, img_rgb):
        """Pap smear — cervical cell classification."""
        ids, feats = self._extract_features(masks, img_rgb)
        results = {}
        for cid, f in zip(ids, feats):
            h_mean, h_std, e_mean, ne_ratio, \
            area, ecc, sol, blue_frac, roundness, _ = f

            if sol < 0.6:
                ct = "Dyskeratotic"
            elif ecc > 0.78:
                ct = "Koilocyte"
            elif area < 80:
                ct = "Parabasal"
            elif ne_ratio > 1.8:
                ct = "Dyskeratotic"
            elif area > 400:
                ct = "Superficial"
            elif 0.5 < roundness <= 0.85:
                ct = "Intermediate"
            else:
                ct = "Metaplastic"
            results[cid] = ct
        return results


# ═════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════
def compute_metrics(pred, gt, iou_thr=0.5):
    if gt is None:
        return None, None, 0, 0, 0
    pred_ids = np.unique(pred[pred > 0])
    true_ids = np.unique(gt[gt > 0])
    if len(true_ids) == 0:
        return 0.0, 1.0, 0, len(pred_ids), 0
    tp = 0; matched = set()
    for pid in pred_ids:
        pm = (pred == pid); best, btid = 0, -1
        for tid in true_ids:
            if tid in matched: continue
            tm   = (gt == tid)
            iou  = np.logical_and(pm,tm).sum() / (np.logical_or(pm,tm).sum()+1e-8)
            if iou > best: best, btid = iou, tid
        if best >= iou_thr:
            tp += 1; matched.add(btid)
    fp = len(pred_ids) - tp; fn = len(true_ids) - tp
    return tp/(tp+fp+fn+1e-8),(fp+fn)/(tp+fn+1e-8),tp,fp,fn


# ═════════════════════════════════════════════════════════════════════
#  MAIN GUI
# ═════════════════════════════════════════════════════════════════════
from PIL import Image, ImageTk

class ImageCropperDialog:
    def __init__(self, parent, arr):
        self.top = tk.Toplevel(parent)
        self.top.title("Large Image Detected — Select Region of Interest")
        self.top.configure(bg="#1E2A38")
        
        # Calculate thumbnail scale to fit within ~500x500
        self.full_h, self.full_w = arr.shape[:2]
        self.scale = max(1, max(self.full_h, self.full_w) // 500)
        
        self.thumb = arr[::self.scale, ::self.scale].copy()
        if self.thumb.max() <= 1.0:
            self.thumb = (self.thumb * 255)
        self.thumb = self.thumb.astype(np.uint8)
        
        self.img_pi = Image.fromarray(self.thumb)
        self.tk_img = ImageTk.PhotoImage(self.img_pi)
        
        lbl = tk.Label(self.top, text=f"Image is {self.full_w}x{self.full_h}. Select a region to analyze (click and drag):", fg="white", bg="#1E2A38", font=("Segoe UI", 12))
        lbl.pack(pady=10)
        
        # Pack button at bottom FIRST to guarantee visibility
        btn_frame = tk.Frame(self.top, bg="#1E2A38")
        btn_frame.pack(side="bottom", fill="x", pady=10)
        
        self.btn = tk.Button(btn_frame, text="Confirm Crop & Load", command=self.top.destroy, bg="#008080", fg="white", font=("Segoe UI", 11, "bold"), state="disabled")
        self.btn.pack()
        
        self.canvas = tk.Canvas(self.top, width=self.thumb.shape[1], height=self.thumb.shape[0], bg="black", cursor="cross")
        self.canvas.pack(pady=10, expand=True)
        
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        
        self.rect_id = None
        self.start_x = None
        self.start_y = None
        self.roi = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.top.grab_set()
        self.top.wait_window()
        
    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="yellow", width=2, dash=(4,4))
        
    def on_drag(self, event):
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)
        
    def on_release(self, event):
        end_x, end_y = event.x, event.y
        x1, x2 = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])
        
        x1, x2 = max(0, int(x1 * self.scale)), min(int(x2 * self.scale), self.full_w)
        y1, y2 = max(0, int(y1 * self.scale)), min(int(y2 * self.scale), self.full_h)
        
        if x2 > x1 and y2 > y1:
            self.roi = (y1, y2, x1, x2)
            self.btn.config(state="normal", text="✅ Selection Ready — Click to Load")

class CellAnalysisGUI:

    def __init__(self, root):
        self.root = root
        self.root.title(
            "Cellpose-SAM v3 — Cell Classification | Localisation | "
            "Boundary | Separation | Pap Smear | Blood Smear")
        self.root.configure(bg=NAVY)
        self.root.state("zoomed")

        self.img_path    = None
        self.img_array   = None
        self.gt_masks    = None
        self.masks       = None
        self.cell_types  = {}
        self.model_cyto3 = None
        self.model_cpsam = None
        self.classifier  = HEClassifier()

        # ── Cell navigation state ─────────────────────────────────────
        self._props_cache      = []     # list of regionprops from last run
        self._dl_results_cache = {}     # {label: (cls, conf)}
        self._tissue_cache     = {}     # {label: tissue_type_str}
        self._selected_cid     = None   # currently highlighted cell label
        self._show_only_sel    = False  # dim-all-others toggle

        try:
            from cervical_inference import CervicalClassifier
            self.cervical_ai = CervicalClassifier()
        except Exception as e:
            self.cervical_ai = None
            print(f"Cervical AI not loaded: {e}")

        self.mode_var     = tk.StringVar(value="cpsam")
        self.analysis_var = tk.StringVar(value="tissue")

        self._build_ui()
        self._log("v3 ready — H&E colour deconvolution classifier loaded")
        self._log(f"GPU: {torch.cuda.is_available()}")

    # ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        ban = tk.Frame(self.root, bg=NAVY, height=50)
        ban.pack(fill="x")
        tk.Label(ban, text="Cellpose-SAM  ·  Cell Classification + Analysis  v3.0",
                 font=("Segoe UI",14,"bold"), fg=TEAL, bg=NAVY
                 ).pack(side="left", padx=14, pady=8)
        tk.Label(ban,
                 text="H&E Colour Deconvolution  ·  Classification  ·  "
                      "Localisation  ·  Boundary  ·  Overlap Separation  ·  Pap/Blood",
                 font=("Segoe UI",8), fg=DGRAY, bg=NAVY
                 ).pack(side="right", padx=14)

        body = tk.Frame(self.root, bg=NAVY)
        body.pack(fill="both", expand=True)
        self._build_left(body)
        self._build_right(body)

    # ── LEFT PANEL ───────────────────────────────────────────────────
    def _build_left(self, parent):
        outer_lf = tk.Frame(parent, bg=PANEL, width=325)
        outer_lf.pack(side="left", fill="y", padx=(7,3), pady=7)
        outer_lf.pack_propagate(False)

        canvas = tk.Canvas(outer_lf, bg=PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer_lf, orient="vertical", command=canvas.yview)
        
        # Pack scrollbar FIRST so canvas doesn't push it out
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        lf = tk.Frame(canvas, bg=PANEL)
        canvas_win = canvas.create_window((0, 0), window=lf, anchor="nw")
        
        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_win, width=event.width)
        canvas.bind('<Configure>', _on_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        def _on_mousewheel(event):
            if canvas.bbox("all") and canvas.bbox("all")[3] > canvas.winfo_height():
                if hasattr(event, 'delta') and event.delta != 0:
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                elif hasattr(event, 'num'):
                    if event.num == 4: canvas.yview_scroll(-1, "units")
                    elif event.num == 5: canvas.yview_scroll(1, "units")

        def _bind_mouse(e):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)
        def _unbind_mouse(e):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
            
        outer_lf.bind("<Enter>", _bind_mouse)
        outer_lf.bind("<Leave>", _unbind_mouse)

        def sec(t, color=TEAL):
            tk.Label(lf,text=t,font=("Segoe UI",10,"bold"),fg=color,bg=PANEL
                     ).pack(anchor="w",padx=11,pady=(6,1))
            tk.Frame(lf,bg=color,height=1).pack(fill="x",padx=11,pady=(0,3))

        def btn(t, cmd, bg=TEAL, fg=NAVY):
            b = tk.Button(lf,text=t,command=cmd,bg=bg,fg=fg,
                          font=("Segoe UI",9,"bold"),relief="flat",
                          cursor="hand2",padx=5,pady=2)
            b.pack(fill="x",padx=11,pady=1)
            return b

        # 1 — Load
        sec("1.  Load Image")
        btn("📂  Browse Image…", self._load_image)
        self.lbl_file = tk.Label(lf,text="No file",font=("Segoe UI",8),
                                  fg=DGRAY,bg=PANEL,wraplength=270)
        self.lbl_file.pack(anchor="w",padx=11)
        btn("📂  Load Ground Truth (optional)",self._load_gt,bg=LGRAY,fg=WHITE)
        self.lbl_gt = tk.Label(lf,text="No GT",font=("Segoe UI",8),
                                fg=DGRAY,bg=PANEL,wraplength=270)
        self.lbl_gt.pack(anchor="w",padx=11)

        # 2 — Model
        sec("2.  Model")
        models = [
            ("cpsam", "Cellpose-SAM", "(paper model — recommended)", TEAL),
            ("cyto3", "Cellpose cyto3", "(baseline)", AMBER)
        ]
        for val, title, desc, c in models:
            tk.Radiobutton(lf,text=title,variable=self.mode_var,value=val,
                           font=("Segoe UI",9,"bold"),fg=c,bg=PANEL,
                           selectcolor=LGRAY,activebackground=PANEL
                           ).pack(anchor="w",padx=15,pady=(2,0))
            tk.Label(lf,text=desc,font=("Segoe UI",8),fg=DGRAY,bg=PANEL).pack(anchor="w",padx=35,pady=(0,2))

        # 3 — Analysis mode
        sec("3.  Image Type")
        modes = [
            ("tissue", "H&E Tissue", "Neoplastic / Inflammatory / Epithelial…"),
            ("blood",  "Blood Smear", "WBC subtypes / RBC / Platelet"),
            ("pap",    "Pap Smear", "Cervical — Bethesda system"),
        ]
        for val, title, desc in modes:
            tk.Radiobutton(lf,text=title,variable=self.analysis_var,value=val,
                           font=("Segoe UI",9,"bold"),fg=WHITE,bg=PANEL,
                           selectcolor=LGRAY,activebackground=PANEL
                           ).pack(anchor="w",padx=15,pady=(2,0))
            tk.Label(lf,text=desc,font=("Segoe UI",8),fg=DGRAY,bg=PANEL).pack(anchor="w",padx=35,pady=(0,2))

        # info
        self.mode_info = tk.Label(lf,text="",font=("Segoe UI",7),
                                   fg=TEAL2,bg=LGRAY,wraplength=270,
                                   justify="left",padx=5,pady=4)
        self.mode_info.pack(fill="x",padx=11,pady=3)
        self.analysis_var.trace_add("write",self._refresh_info)
        self._refresh_info()

        # 4 — Parameters
        sec("4.  Parameters")
        params = [
            ("Cell Diameter (px):",   "diam_var","30",
             "30 = general tissue\n18 = small nuclei (H&E)"),
            ("Flow Threshold:",        "flow_var","0.4",
             "0.4 default  |  0.2 = more cells  |  0.8 = stricter"),
            ("Cell Prob Threshold:","prob_var","0.0",
             "0.0 = all  |  0.5 = confident only"),
        ]
        for lbl,var,default,tip in params:
            tk.Label(lf,text=lbl,font=("Segoe UI",9),fg=WHITE,bg=PANEL
                     ).pack(anchor="w",padx=11)
            setattr(self,var,tk.StringVar(value=default))
            ent = tk.Entry(lf,textvariable=getattr(self,var),
                           font=("Segoe UI",9),width=10,
                           bg=LGRAY,fg=WHITE,insertbackground=WHITE)
            ent.pack(anchor="w",padx=11,pady=(0,1))
            tk.Label(lf,text=tip,font=("Segoe UI",7),fg=DGRAY,bg=PANEL,
                     wraplength=265,justify="left"
                     ).pack(anchor="w",padx=11,pady=(0,4))

        # 5 — Run
        sec("5.  Run")
        btn("▶  RUN FULL ANALYSIS",self._run_full,bg=TEAL,fg=NAVY)
        self.prog = ttk.Progressbar(lf,mode="indeterminate",length=270)
        self.prog.pack(padx=11,pady=3)
        self.status_lbl = tk.Label(lf,text="Idle",font=("Segoe UI",9),
                                    fg=TEAL2,bg=PANEL)
        self.status_lbl.pack(anchor="w",padx=11)

        # 6 — Big count
        sec("6.  Cell Count",color=AMBER)
        cf = tk.Frame(lf,bg=LGRAY)
        cf.pack(fill="x",padx=11,pady=3)
        self.count_lbl = tk.Label(cf,text="—",font=("Segoe UI",38,"bold"),
                                   fg=AMBER,bg=LGRAY)
        self.count_lbl.pack(side="left",padx=14,pady=3)
        self.count_sub = tk.Label(cf,text="cells\ndetected",
                                   font=("Segoe UI",10),fg=DGRAY,bg=LGRAY,
                                   justify="left")
        self.count_sub.pack(side="left")

        # 7 — Export
        sec("7.  Export")
        btn("💾  Save Current Tab",   self._save_vis, bg=GREEN,fg=WHITE)
        btn("📄  Export CSV (all data)",self._export_csv,bg=LGRAY,fg=WHITE)
        btn("📄  Export Boundaries",  self._export_bnd,bg=LGRAY,fg=WHITE)

        # log
        sec("Log")
        self.logbox = tk.Text(lf,height=3,bg=LGRAY,fg=DGRAY,
                               font=("Consolas",7),relief="flat",
                               state="disabled",wrap="word")
        self.logbox.pack(fill="x",padx=11,pady=(0,8))

    # ── RIGHT PANEL ──────────────────────────────────────────────────
    def _build_right(self, parent):
        rf = tk.Frame(parent,bg=NAVY)
        rf.pack(side="left",fill="both",expand=True,padx=(3,7),pady=7)

        # tab bar
        tb = tk.Frame(rf,bg=NAVY); tb.pack(fill="x")
        self.tab_btns   = {}
        self.tab_frames = {}
        tabs = [
            ("seg",    "🔬 Segmentation"),
            ("class",  "🏷️ Classification"),
            ("local",  "📍 Localisation"),
            ("bnd",    "✏️ Boundaries"),
            ("sep",    "✂️ Separation"),
            ("pap",    "🔵 Pap Smear"),
            ("blood",  "🩸 Blood Smear"),
        ]
        for key,label in tabs:
            b = tk.Button(tb,text=label,
                          command=lambda k=key:self._tab(k),
                          font=("Segoe UI",9,"bold"),relief="flat",
                          cursor="hand2",padx=9,pady=5)
            b.pack(side="left",padx=2)
            self.tab_btns[key] = b

        ct = tk.Frame(rf,bg=NAVY); ct.pack(fill="both",expand=True)
        for key,_ in tabs:
            f = tk.Frame(ct,bg=NAVY)
            f.place(relwidth=1,relheight=1)
            self.tab_frames[key] = f

        self._build_seg_tab()
        self._build_class_tab()
        self._build_local_tab()
        self._build_bnd_tab()
        self._build_sep_tab()
        self._build_pap_tab()
        self._build_blood_tab()
        self._tab("seg")

    def _tab(self, key):
        for k,f in self.tab_frames.items(): f.lower()
        self.tab_frames[key].lift()
        for k,b in self.tab_btns.items():
            b.config(bg=TEAL if k==key else LGRAY,
                     fg=NAVY if k==key else WHITE)

    # ── TAB BUILDERS ─────────────────────────────────────────────────
    def _make_fig(self, parent, rows, cols, title="",
                  col_titles=None, col_colors=None, figsize=(12,6.8)):
        if title:
            tk.Label(parent,text=title,font=("Segoe UI",10,"bold"),
                     fg=WHITE,bg=NAVY).pack(anchor="w")
        fig = Figure(figsize=figsize,facecolor=NAVY)
        fig.subplots_adjust(left=0.02,right=0.98,top=0.93,
                             bottom=0.04,wspace=0.04,hspace=0.15)
        if rows==1 and cols==1:
            axes = np.array([[fig.add_subplot(1,1,1)]])
        else:
            axes = np.array(fig.subplots(rows,cols))
            if axes.ndim==1: axes = axes.reshape(1,-1)
        for ax in axes.flat:
            ax.set_facecolor(LGRAY); ax.set_xticks([]); ax.set_yticks([])
        if col_titles:
            for i,(t,c) in enumerate(zip(col_titles,
                                          col_colors or [DGRAY]*len(col_titles))):
                axes[0,i].set_title(t,color=c,fontsize=9,fontweight="bold")
        canvas = FigureCanvasTkAgg(fig,master=parent)
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both",expand=True)
        return fig, axes, canvas

    def _build_seg_tab(self):
        f = self.tab_frames["seg"]
        self.seg_fig,self.seg_ax,self.seg_cv = self._make_fig(
            f,1,3,"Segmentation Results",
            ["Input Image","Cell Masks + Boundaries","Cell IDs Overlay"],
            [DGRAY,TEAL,AMBER])

    def _build_class_tab(self):
        f = self.tab_frames["class"]
        tk.Label(f,text="Cell Classification — H&E colour deconvolution assigns each cell a type",
                 font=("Segoe UI",10,"bold"),fg=WHITE,bg=NAVY).pack(anchor="w")
        fig = Figure(figsize=(12,6.8),facecolor=NAVY)
        fig.subplots_adjust(left=0.02,right=0.98,top=0.93,
                             bottom=0.04,wspace=0.12)
        gs = fig.add_gridspec(1,3,width_ratios=[3,1.4,1.4])
        self.cls_img_ax  = fig.add_subplot(gs[0])
        self.cls_bar_ax  = fig.add_subplot(gs[1])
        self.cls_pie_ax  = fig.add_subplot(gs[2])
        for ax in [self.cls_img_ax,self.cls_bar_ax,self.cls_pie_ax]:
            ax.set_facecolor(LGRAY)
        self.cls_img_ax.set_xticks([]); self.cls_img_ax.set_yticks([])
        # legend strip packed first to the bottom
        self.cls_leg_frame = tk.Frame(f,bg=LGRAY)
        self.cls_leg_frame.pack(side="bottom", fill="x", pady=2)
        
        fig_frame = tk.Frame(f, bg=NAVY)
        fig_frame.pack(side="top", fill="both", expand=True)

        self.cls_fig = fig
        self.cls_cv  = FigureCanvasTkAgg(fig,master=fig_frame)
        toolbar1 = NavigationToolbar2Tk(self.cls_cv, fig_frame)
        toolbar1.update()
        self.cls_cv.get_tk_widget().pack(fill="both",expand=True)

    def _build_local_tab(self):
        f = self.tab_frames["local"]

        # ── top label ────────────────────────────────────────────────
        tk.Label(f, text="Cell Localisation — click any row to navigate  │  "
                         "Jump-To-Cell for direct access",
                 font=("Segoe UI",10,"bold"), fg=WHITE, bg=NAVY
                 ).pack(anchor="w", padx=6, pady=(4,0))

        body = tk.Frame(f, bg=NAVY)
        body.pack(fill="both", expand=True)

        # ── LEFT: matplotlib canvas ───────────────────────────────────
        canvas_frame = tk.Frame(body, bg=NAVY)
        canvas_frame.pack(side="left", fill="both", expand=True)

        fig = Figure(figsize=(8, 6.5), facecolor=NAVY)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)
        self.loc_ax = fig.add_subplot(111)
        self.loc_ax.set_facecolor(LGRAY)
        self.loc_ax.set_xticks([]); self.loc_ax.set_yticks([])
        self.loc_fig = fig
        self.loc_cv  = FigureCanvasTkAgg(fig, master=canvas_frame)
        NavigationToolbar2Tk(self.loc_cv, canvas_frame).update()
        self.loc_cv.get_tk_widget().pack(fill="both", expand=True)

        # ── RIGHT: table + preview panel ─────────────────────────────
        right = tk.Frame(body, bg=LGRAY, width=460)
        right.pack(side="left", fill="y", padx=3)
        right.pack_propagate(False)

        # Jump-to-cell controls
        jump_row = tk.Frame(right, bg=LGRAY)
        jump_row.pack(fill="x", padx=6, pady=4)
        tk.Label(jump_row, text="Jump to Cell ID:", font=("Segoe UI",9),
                 fg=AMBER, bg=LGRAY).pack(side="left")
        self._jump_var = tk.StringVar()
        jump_entry = tk.Entry(jump_row, textvariable=self._jump_var,
                              width=7, bg=PANEL, fg=WHITE,
                              insertbackground=WHITE, font=("Segoe UI",9))
        jump_entry.pack(side="left", padx=4)
        jump_entry.bind("<Return>", lambda e: self._jump_to_cell())
        tk.Button(jump_row, text="Go", command=self._jump_to_cell,
                  bg=TEAL, fg=NAVY, font=("Segoe UI",8,"bold"),
                  relief="flat", padx=6).pack(side="left")

        # Show-only-selected toggle
        self._solo_var = tk.BooleanVar(value=False)
        tk.Checkbutton(jump_row, text="Solo", variable=self._solo_var,
                       command=self._on_solo_toggle,
                       bg=LGRAY, fg=DGRAY, selectcolor=PANEL,
                       activebackground=LGRAY, font=("Segoe UI",8)
                       ).pack(side="left", padx=6)

        # Table heading
        tk.Label(right, text="Cell Coordinates & AI Class",
                 font=("Segoe UI",10,"bold"), fg=AMBER, bg=LGRAY
                 ).pack(pady=(0,2))

        # Treeview + scrollbar
        tree_frame = tk.Frame(right, bg=LGRAY)
        tree_frame.pack(fill="both", expand=True)

        cols = ("ID", "Cx", "Cy", "Area", "Type", "DL Class", "Conf(%)")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                 height=18, selectmode="browse")
        col_widths = {"ID":40, "Cx":46, "Cy":46, "Area":52,
                      "Type":72, "DL Class":68, "Conf(%)":58}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=col_widths.get(c, 50), anchor="center")

        # Selected-row highlight tag
        self.tree.tag_configure("selected_row",
                                background="#2a3f1a", foreground=AMBER)

        sb = ttk.Scrollbar(tree_frame, orient="vertical",
                           command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_cell_select)

        # ── PREVIEW PANEL (below table) ───────────────────────────────
        prev = tk.LabelFrame(right, text=" Selected Cell Preview ",
                             bg=LGRAY, fg=TEAL,
                             font=("Segoe UI",9,"bold"))
        prev.pack(fill="x", padx=4, pady=4)

        prev_inner = tk.Frame(prev, bg=LGRAY)
        prev_inner.pack(fill="x")

        # Thumbnail canvas
        self._preview_canvas = tk.Canvas(prev_inner, width=110, height=110,
                                         bg=PANEL, highlightthickness=1,
                                         highlightbackground=TEAL)
        self._preview_canvas.pack(side="left", padx=6, pady=6)
        self._preview_photo = None  # keep reference

        # Info labels
        info_f = tk.Frame(prev_inner, bg=LGRAY)
        info_f.pack(side="left", fill="both", expand=True, padx=4)

        self._prev_labels = {}
        for row_i, (key, lbl) in enumerate([
            ("cell_id",   "Cell ID"),
            ("dl_class",  "DL Class"),
            ("conf",      "Confidence"),
            ("area",      "Area (px)"),
            ("tissue",    "Tissue Type"),
        ]):
            tk.Label(info_f, text=f"{lbl}:",
                     font=("Segoe UI",8,"bold"), fg=DGRAY, bg=LGRAY,
                     anchor="w").grid(row=row_i, column=0, sticky="w", pady=1)
            val_lbl = tk.Label(info_f, text="—",
                               font=("Segoe UI",9), fg=WHITE, bg=LGRAY,
                               anchor="w")
            val_lbl.grid(row=row_i, column=1, sticky="w", padx=6, pady=1)
            self._prev_labels[key] = val_lbl

    # ── CELL NAVIGATION HANDLERS ─────────────────────────────────────

    def _redraw_loc_base(self):
        """Draw all cells on the localisation axis using the cached props."""
        img   = self.img_array
        props = self._props_cache
        if img is None or not props:
            return

        ax = self.loc_ax
        ax.cla()
        ax.set_facecolor(LGRAY)
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(np.clip(img, 0, 1))

        solo = self._show_only_sel
        sel  = self._selected_cid

        for p in props:
            cy, cx = p.centroid
            mn_r, mn_c, mx_r, mx_c = p.bbox
            ct  = self._tissue_cache.get(p.label, "Unknown")
            col = TISSUE_TYPES.get(ct, {}).get("color", TEAL)

            # In Solo mode dim everything except the selected cell
            if solo and sel is not None and p.label != sel:
                alpha_dot  = 0.08
                alpha_box  = 0.06
            else:
                alpha_dot  = 0.88
                alpha_box  = 0.70

            ax.plot(cx, cy, "o", ms=2.5, color=col, alpha=alpha_dot)
            rect = plt.Rectangle(
                (mn_c, mn_r), mx_c - mn_c, mx_r - mn_r,
                lw=0.7, edgecolor=col, facecolor="none", alpha=alpha_box)
            ax.add_patch(rect)

        # Draw selection highlight on top
        if sel is not None:
            self._draw_highlight(ax, sel)

        n = len(props)
        ax.set_title(
            f"Localisation — {n} cells   "
            "(click row to navigate  |  Solo = dim others)",
            color=TEAL, fontsize=9, fontweight="bold")
        self.loc_cv.draw()

    def _draw_highlight(self, ax, cell_id):
        """Add a thick yellow bounding box + red crosshair for cell_id."""
        for p in self._props_cache:
            if p.label != cell_id:
                continue
            cy, cx = p.centroid
            mn_r, mn_c, mx_r, mx_c = p.bbox
            w = mx_c - mn_c
            h = mx_r - mn_r
            # Thick yellow selection box
            ax.add_patch(plt.Rectangle(
                (mn_c - 2, mn_r - 2), w + 4, h + 4,
                lw=2.8, edgecolor=AMBER, facecolor="none", alpha=1.0,
                zorder=10))
            # Bright red inner outline
            ax.add_patch(plt.Rectangle(
                (mn_c, mn_r), w, h,
                lw=1.2, edgecolor=RED, facecolor="none", alpha=0.85,
                zorder=11))
            # Crosshair at centroid
            cross = 12
            ax.plot([cx - cross, cx + cross], [cy, cy],
                    color=AMBER, lw=1.8, alpha=0.95, zorder=12)
            ax.plot([cx, cx], [cy - cross, cy + cross],
                    color=AMBER, lw=1.8, alpha=0.95, zorder=12)
            ax.plot(cx, cy, "o", ms=5, color=RED, alpha=1.0, zorder=13)
            break

    def _zoom_to_cell(self, cell_id):
        """Pan-zoom the localisation axis so the selected cell fills ~40% of the view."""
        if self.img_array is None:
            return
        H, W = self.img_array.shape[:2]
        for p in self._props_cache:
            if p.label != cell_id:
                continue
            cy, cx = p.centroid
            mn_r, mn_c, mx_r, mx_c = p.bbox
            cell_w = max(mx_c - mn_c, mx_r - mn_r, 60)
            pad    = cell_w * 2.5            # context padding around the cell
            x0 = max(0,  cx - pad)
            x1 = min(W,  cx + pad)
            y0 = max(0,  cy - pad)
            y1 = min(H,  cy + pad)
            self.loc_ax.set_xlim(x0, x1)
            self.loc_ax.set_ylim(y1, y0)   # y-axis inverted for images
            self.loc_cv.draw()
            break

    def _update_preview_panel(self, cell_id):
        """Render the cropped thumbnail and fill the info labels for cell_id."""
        from PIL import Image as _PIL, ImageTk

        # Clear old labels
        for lbl in self._prev_labels.values():
            lbl.config(text="—", fg=WHITE)
        self._preview_canvas.delete("all")

        if self.img_array is None:
            return

        for p in self._props_cache:
            if p.label != cell_id:
                continue

            # ── thumbnail ──────────────────────────────────────────
            mn_r, mn_c, mx_r, mx_c = p.bbox
            crop = self.img_array[mn_r:mx_r, mn_c:mx_c].copy()
            crop = np.clip(crop, 0, 1)
            crop_u8 = (crop * 255).astype(np.uint8)
            pil_img = _PIL.fromarray(crop_u8)
            pil_img = pil_img.resize((108, 108), _PIL.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(pil_img)
            self._preview_canvas.create_image(
                54, 54, anchor="center", image=self._preview_photo)

            # ── info text ──────────────────────────────────────────
            dl_cls, dl_conf = self._dl_results_cache.get(cell_id, ("-", 0.0))
            tissue = self._tissue_cache.get(cell_id, "—")
            conf_str = f"{dl_conf:.1f}%" if dl_conf > 0 else "—"

            # Colour-code the DL class
            cls_colors = {"HSIL": "#E24B4A", "LSIL": "#F0B429",
                          "ASCUS": "#7C6AF5", "NILM": "#00A896"}
            cls_col = cls_colors.get(dl_cls, WHITE)

            self._prev_labels["cell_id" ].config(text=str(cell_id))
            self._prev_labels["dl_class"].config(text=dl_cls, fg=cls_col)
            self._prev_labels["conf"    ].config(text=conf_str)
            self._prev_labels["area"    ].config(text=str(p.area))
            self._prev_labels["tissue"  ].config(text=tissue)
            break

    def _on_cell_select(self, event=None):
        """Called when the user clicks a Treeview row."""
        sel = self.tree.selection()
        if not sel:
            return
        try:
            cell_id = int(sel[0])
        except (ValueError, IndexError):
            return

        # Remove highlight tag from all rows, apply to selected
        for iid in self.tree.get_children():
            self.tree.item(iid, tags=())
        self.tree.item(str(cell_id), tags=("selected_row",))

        self._selected_cid = cell_id
        self._redraw_loc_base()
        self._zoom_to_cell(cell_id)
        self._update_preview_panel(cell_id)

    def _jump_to_cell(self):
        """Read the Jump-To-Cell entry and navigate there."""
        try:
            cell_id = int(self._jump_var.get().strip())
        except ValueError:
            messagebox.showwarning("Jump To Cell",
                                   "Enter a valid integer Cell ID.")
            return

        # Check it exists
        iid = str(cell_id)
        if iid not in self.tree.get_children():
            messagebox.showwarning("Jump To Cell",
                                   f"Cell ID {cell_id} not found in this image.")
            return

        # Select the row and scroll the table to it
        self.tree.selection_set(iid)
        self.tree.see(iid)
        self._on_cell_select()

    def _on_solo_toggle(self):
        """Toggle the 'Solo' (show only selected) mode and redraw."""
        self._show_only_sel = self._solo_var.get()
        self._redraw_loc_base()
        if self._selected_cid is not None:
            self._zoom_to_cell(self._selected_cid)

    def _build_bnd_tab(self):
        f = self.tab_frames["bnd"]
        self.bnd_fig,self.bnd_ax,self.bnd_cv = self._make_fig(
            f,1,3,"Cell Boundary Extraction",
            ["Original","Extracted Boundaries","Overlay"],
            [DGRAY,TEAL,AMBER])

    def _build_sep_tab(self):
        f = self.tab_frames["sep"]
        self.sep_fig,self.sep_ax,self.sep_cv = self._make_fig(
            f,1,3,"Overlap Cell Separation (Watershed)",
            ["Original Masks","After Watershed Separation","Separation Lines"],
            [AMBER,TEAL,RED])
        self.sep_stat = tk.Label(f,text="Run analysis to see overlap statistics",
                                  font=("Segoe UI",10),fg=TEAL2,bg=NAVY)
        self.sep_stat.pack()

    def _build_pap_tab(self):
        f = self.tab_frames["pap"]
        info = tk.Frame(f,bg=LGRAY); info.pack(fill="x",pady=(0,4))
        tk.Label(info,text="🔵  PAP SMEAR ANALYSIS — Cervical Cancer Screening",
                 font=("Segoe UI",11,"bold"),fg=TEAL,bg=LGRAY
                 ).pack(side="left",padx=11,pady=5)
        tk.Label(info,
            text="Papanicolaou (Pap) test: cervical cells collected, "
                 "stained & classified. AI detects early abnormalities. "
                 "Bethesda system: Normal (Superficial/Intermediate/Parabasal) "
                 "vs Abnormal (Koilocyte/Dyskeratotic/Metaplastic).",
            font=("Segoe UI",8),fg=DGRAY,bg=LGRAY,wraplength=800,justify="left"
            ).pack(side="left",padx=6)
        # legend packed first to the bottom
        leg = tk.Frame(f,bg=LGRAY); leg.pack(side="bottom", fill="x", pady=1)
        for i, (ct,info_d) in enumerate(PAP_TYPES.items()):
            tk.Label(leg,text=f"■ {ct} — {info_d['meaning']}",
                     font=("Segoe UI",9),fg=info_d["color"],bg=LGRAY
                     ).grid(row=i//3, column=i%3, sticky="w", padx=10, pady=2)
                     
        fig_frame = tk.Frame(f, bg=NAVY)
        fig_frame.pack(side="top", fill="both", expand=True)
        self.pap_fig,self.pap_ax,self.pap_cv = self._make_fig(
            fig_frame,1,3,"Pap Smear Classification",
            ["Input","Classified Cells","Distribution"],
            [DGRAY,TEAL,AMBER],figsize=(12,5.8))

    def _build_blood_tab(self):
        f = self.tab_frames["blood"]
        info = tk.Frame(f,bg=LGRAY); info.pack(fill="x",pady=(0,4))
        tk.Label(info,text="🩸  BLOOD SMEAR ANALYSIS — WBC / RBC / Platelet",
                 font=("Segoe UI",11,"bold"),fg=RED,bg=LGRAY
                 ).pack(side="left",padx=11,pady=5)
        tk.Label(info,
            text="Blood smear: identifies cell types in a blood sample. "
                 "WBC (white blood cells) subtypes: Neutrophil (immune defence), "
                 "Lymphocyte (adaptive immunity), Monocyte (inflammation), "
                 "Eosinophil (allergy). RBC (red blood cells). Platelets (clotting).",
            font=("Segoe UI",8),fg=DGRAY,bg=LGRAY,wraplength=800,justify="left"
            ).pack(side="left",padx=6)
        leg = tk.Frame(f,bg=LGRAY); leg.pack(side="bottom", fill="x", pady=1)
        for i, (ct,info_d) in enumerate(BLOOD_TYPES.items()):
            tk.Label(leg,text=f"■ {ct} — {info_d['meaning']}",
                     font=("Segoe UI",9),fg=info_d["color"],bg=LGRAY
                     ).grid(row=i//3, column=i%3, sticky="w", padx=10, pady=2)

        fig_frame = tk.Frame(f, bg=NAVY)
        fig_frame.pack(side="top", fill="both", expand=True)
        self.bld_fig,self.bld_ax,self.bld_cv = self._make_fig(
            fig_frame,1,3,"Blood Smear Classification",
            ["Input","Classified Cells","Distribution"],
            [DGRAY,RED,AMBER],figsize=(12,5.8))

    # ── FILE LOADING ─────────────────────────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select microscopy image",
            filetypes=[("Images","*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                       ("All","*.*")])
        if not path: return
        try:
            ext = os.path.splitext(path)[1].lower()

            # ── Format-aware loader ──────────────────────────────────────
            # TIFF / multi-page WSI  → tifffile  (handles BigTIFF, OME-TIFF)
            # BMP / PNG / JPG / JPEG → PIL        (handles all standard formats)
            if ext in (".tif", ".tiff"):
                arr = tifffile.imread(path)
            else:
                from PIL import Image as _PIL
                with _PIL.open(path) as _im:
                    _im = _im.convert("RGB")   # normalise RGBA/P/L → RGB
                    arr = np.array(_im, dtype=np.uint8)

            # ── Channel normalisation (handles CYX, grayscale, RGBA) ──
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            elif arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.moveaxis(arr, 0, -1)
                if arr.shape[2] == 1:
                    arr = np.concatenate([arr]*3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]

            # ── WSI Large Image Handler ──────────────────────────────────
            if arr.shape[0] * arr.shape[1] > 10_000_000:  # > ~10 megapixels
                dialog = ImageCropperDialog(self.root, arr)
                if dialog.roi:
                    y1, y2, x1, x2 = dialog.roi
                    arr = arr[y1:y2, x1:x2].copy()
                else:
                    return  # User cancelled

            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr /= arr.max()

            self.img_array = arr
            self.img_path  = path
            short = os.path.basename(path)
            self.lbl_file.config(text=short)
            self._log(f"Loaded [{ext.lstrip('.').upper()}]: {short}  "
                      f"({arr.shape[0]}×{arr.shape[1]}px)")
            self._show_input()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _load_gt(self):
        path = filedialog.askopenfilename(
            filetypes=[("Mask","*.npy *.tif *.tiff"),("All","*.*")])
        if not path: return
        try:
            if path.endswith(".npy"):
                d = np.load(path,allow_pickle=True)
                self.gt_masks = d.item().get("masks",None) \
                    if d.dtype==object else d
            else:
                self.gt_masks = tifffile.imread(path).astype(np.int32)
            n = int(self.gt_masks.max())
            self.lbl_gt.config(
                text=f"{os.path.basename(path)} ({n} cells)")
            self._log(f"GT loaded: {n} cells")
        except Exception as e:
            messagebox.showerror("Error",str(e))

    def _show_input(self):
        img = self.img_array
        for ax in self.seg_ax.flat:
            ax.cla(); ax.set_facecolor(LGRAY)
            ax.set_xticks([]); ax.set_yticks([])
        self.seg_ax[0,0].imshow(np.clip(img,0,1))
        self.seg_ax[0,0].set_title("Input Image",color=DGRAY,
                                    fontsize=9,fontweight="bold")
        for ax in [self.seg_ax[0,1],self.seg_ax[0,2]]:
            ax.text(0.5,0.5,"Run analysis",ha="center",va="center",
                    color=DGRAY,fontsize=10,transform=ax.transAxes)
        self.seg_cv.draw()

    # ── MODEL LOADING ────────────────────────────────────────────────
    def _ensure(self, which):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if which=="cpsam" and self.model_cpsam is None:
            self._status("Loading Cellpose-SAM weights…")
            self.model_cpsam = models.CellposeModel(
                model_type="cpsam",device=dev)
        elif which=="cyto3" and self.model_cyto3 is None:
            self._status("Loading cyto3 weights…")
            self.model_cyto3 = models.CellposeModel(
                model_type="cyto3",device=dev)

    # ── SEGMENTATION ─────────────────────────────────────────────────
    def _segment(self):
        which = self.mode_var.get()
        self._ensure(which)
        m = self.model_cpsam if which=="cpsam" else self.model_cyto3
        img = self.img_array
        gray = (0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2])
        
        # [NEW] CLAHE for stability and contrast consistency across stains
        gray = equalize_adapthist(gray, clip_limit=0.03)
        
        g8   = (gray*255).astype(np.uint8)
        diam = float(self.diam_var.get() or 30)
        flow = float(self.flow_var.get() or 0.4)
        prob = float(self.prob_var.get() or 0.0)
        
        # [NEW] Adaptive Parameter Tuning
        analysis_mode = self.analysis_var.get()
        if analysis_mode == "pap":
            flow = min(flow + 0.1, 0.9)  # Higher flow to prevent overmerging loose cervical cells
            prob = prob - 0.5            # More sensitive to faint cytoplasm
        elif analysis_mode == "blood":
            flow = max(flow - 0.1, 0.1)  # Lower flow for dense, round blood cells
            
        t0 = time.time()
        masks,_,_ = m.eval(g8,diameter=diam,channels=[0,0],
                            flow_threshold=flow,cellprob_threshold=prob)
        return masks, time.time()-t0, diam

    # ── POST-PROCESSING ENGINE ───────────────────────────────────────
    def _post_process_masks(self, masks, diam):
        """IoU merging, Centroid distance filtering, and Area thresholding."""
        min_area = (diam * diam) * 0.15
        props = regionprops(masks)
        
        valid_props = [p for p in props if p.area >= min_area]
        if not valid_props: return masks

        def get_iou(bb1, bb2):
            r_min = max(bb1[0], bb2[0]); c_min = max(bb1[1], bb2[1])
            r_max = min(bb1[2], bb2[2]); c_max = min(bb1[3], bb2[3])
            if r_max <= r_min or c_max <= c_min: return 0.0
            inter = (r_max - r_min) * (c_max - c_min)
            area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
            area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
            return inter / float(area1 + area2 - inter)

        n = len(valid_props)
        parent = {p.label: p.label for p in valid_props}
        
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i); root_j = find(j)
            if root_i != root_j: parent[root_i] = root_j

        min_dist = diam * 0.45 
        
        # Build merge graph
        for i in range(n):
            p1 = valid_props[i]
            for j in range(i+1, n):
                p2 = valid_props[j]
                
                # Bounding Box IoU Overlap
                iou = get_iou(p1.bbox, p2.bbox)
                
                # Centroid Proximity
                dy = p1.centroid[0] - p2.centroid[0]
                dx = p1.centroid[1] - p2.centroid[1]
                dist = (dx*dx + dy*dy)**0.5
                
                # Merge if highly overlapping OR centroids are suspiciously close
                if iou > 0.4 or dist < min_dist:
                    union(p1.label, p2.label)
                    
        groups = {}
        for p in valid_props:
            root = find(p.label)
            if root not in groups: groups[root] = []
            groups[root].append(p.label)
            
        # Reconstruct filtered & merged masks
        new_masks = np.zeros_like(masks)
        current_id = 1
        for root, labels in groups.items():
            for l in labels:
                new_masks[masks == l] = current_id
            current_id += 1
            
        return new_masks

    # ── WATERSHED SEPARATION ─────────────────────────────────────────
    def _separate(self, masks):
        binary = (masks>0).astype(np.uint8)
        dist   = distance_transform_edt(binary)
        smooth = gaussian(dist,sigma=2)
        coords = peak_local_max(smooth,min_distance=8,labels=binary)
        lmax   = np.zeros_like(dist,dtype=bool)
        if len(coords)>0: lmax[tuple(coords.T)] = True
        markers,_ = ndi.label(lmax)
        return watershed(-smooth,markers,mask=binary,compactness=0.001)

    # ── CLASSIFY WITH COLOUR MAP ─────────────────────────────────────
    def _colorize(self, masks, cell_types, type_dict, img):
        """Paint each detected cell with its classification colour."""
        out = img.copy()
        for cid,ct in cell_types.items():
            rr,cc = np.where(masks==cid)
            if len(rr)==0: continue
            hex_c = type_dict.get(ct,{}).get("color","#888888")
            r = int(hex_c[1:3],16)/255
            g = int(hex_c[3:5],16)/255
            b = int(hex_c[5:7],16)/255
            out[rr,cc] = 0.25*img[rr,cc]+0.75*np.array([r,g,b])
        return np.clip(out,0,1)

    # ── MAIN RUN ─────────────────────────────────────────────────────
    def _run_full(self):
        if self.img_array is None:
            messagebox.showwarning("No image","Load an image first."); return
        threading.Thread(target=self._run_thread,daemon=True).start()

    def _run_thread(self):
        self._start_prog()
        try:
            # Segment
            self._status("Segmenting…")
            masks, rt, diam = self._segment()
            
            self._status("Post-processing masks (Filtering & Merging)…")
            masks = self._post_process_masks(masks, diam)
            
            self.masks = masks
            n = len(np.unique(masks))-1
            self._log(f"Segmented & Refined: {n} cells in {rt:.2f}s")

            # Classify — tissue
            self._status("Classifying (H&E deconvolution)…")
            tissue_types = self.classifier.classify_tissue(masks,self.img_array)
            self.cell_types = tissue_types

            # Classify — blood + pap (always run for both tabs)
            blood_types = self.classifier.classify_blood(masks,self.img_array)
            pap_types   = self.classifier.classify_pap(masks,self.img_array)

            # Boundaries
            self._status("Extracting boundaries…")
            bnd_img, boundaries = self._get_boundaries(masks)

            # Separation
            self._status("Separating overlaps…")
            sep_masks = self._separate(masks)
            n_sep = len(np.unique(sep_masks))-1

            # Props
            props = regionprops(masks)

            # ── 8-CLASS DEEP LEARNING CLASSIFICATION (Cervical AI) ──
            dl_results = {}
            if getattr(self, 'cervical_ai', None) and self.cervical_ai.model is not None:
                self._status("Running EfficientNet-B0 inference…")
                try:
                    crops = []
                    cids = []
                    for p in props:
                        min_r, min_c, max_r, max_c = p.bbox
                        crop = self.img_array[min_r:max_r, min_c:max_c].copy()
                        if crop.max() <= 1.0: crop = (crop * 255).astype(np.uint8)
                        crops.append(crop)
                        cids.append(p.label)
                    if crops:
                        preds = self.cervical_ai.predict_batch(crops)
                        for cid, (cls_name, conf) in zip(cids, preds):
                            dl_results[cid] = (cls_name, conf)
                except Exception as e:
                    self._log(f"DL Error: {e}")

            self.root.after(0,self._update_all,
                             masks,n,rt,tissue_types,blood_types,pap_types,
                             bnd_img,boundaries,sep_masks,n_sep,props,dl_results)
        except Exception as e:
            self._log(f"ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            self._stop_prog()
            self._status("Done.")

    # ── UPDATE ALL TABS ──────────────────────────────────────────────
    def _update_all(self, masks, n, rt, tissue_types, blood_types,
                    pap_types, bnd_img, boundaries, sep_masks, n_sep, props, dl_results=None):
        img = self.img_array

        # Big count
        self.count_lbl.config(text=str(n))
        self.count_sub.config(
            text=f"cells\n{rt:.2f}s  |  {self.mode_var.get()}")

        # ── SEG TAB ──────────────────────────────────────────────────
        for ax in self.seg_ax.flat:
            ax.cla(); ax.set_facecolor(LGRAY)
            ax.set_xticks([]); ax.set_yticks([])
        self.seg_ax[0,0].imshow(np.clip(img,0,1))
        self.seg_ax[0,0].set_title("Input Image",color=DGRAY,
                                    fontsize=9,fontweight="bold")
        ov = label2rgb(masks,image=img,bg_label=0,alpha=0.42)
        self.seg_ax[0,1].imshow(np.clip(ov,0,1))
        self.seg_ax[0,1].set_title(
            f"Cell Masks — {n} cells found",
            color=TEAL,fontsize=9,fontweight="bold")
        self.seg_ax[0,2].imshow(np.clip(img,0,1))
        for p in props[:300]:
            cy,cx = p.centroid
            self.seg_ax[0,2].text(cx,cy,str(p.label),
                ha="center",va="center",fontsize=4.5,color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.08",fc="black",
                          alpha=0.5,lw=0))
        
        bnd_mask = find_boundaries(masks, mode='inner')
        bnd_rgba = np.zeros((*masks.shape, 4), dtype=np.float32)
        r,g,b = int(TEAL[1:3],16)/255, int(TEAL[3:5],16)/255, int(TEAL[5:7],16)/255
        bnd_rgba[bnd_mask] = [r, g, b, 0.7]
        self.seg_ax[0,2].imshow(bnd_rgba)
        
        self.seg_ax[0,2].set_title("IDs + Boundaries",
            color=AMBER,fontsize=9,fontweight="bold")
        self.seg_cv.draw()

        # ── CLASS TAB ────────────────────────────────────────────────
        self._update_class_tab(masks,tissue_types,TISSUE_TYPES,img,
                                "Tissue Classification — H&E Colour Deconvolution")

        # ── LOCAL TAB ────────────────────────────────────────────────
        # Cache for interactive navigation
        self._props_cache      = list(props)
        self._dl_results_cache = dl_results or {}
        self._tissue_cache     = tissue_types
        self._selected_cid     = None
        self._show_only_sel    = False
        self._solo_var.set(False)

        # Draw base localisation map
        self._redraw_loc_base()

        # Populate table
        for row in self.tree.get_children():
            self.tree.delete(row)
        for p in props:
            cy, cx = p.centroid
            ct = tissue_types.get(p.label, "—")
            dl_cls, dl_conf = (
                dl_results.get(p.label, ("-", 0.0))
                if dl_results else ("-", 0.0)
            )
            conf_str = f"{dl_conf:.1f}%" if dl_conf > 0 else "-"
            # Use the cell label as the item ID so we can look it up instantly
            self.tree.insert("", "end", iid=str(p.label),
                             values=(p.label, f"{cx:.0f}", f"{cy:.0f}",
                                     p.area, ct, dl_cls, conf_str))

        # ── BND TAB ──────────────────────────────────────────────────
        for ax in self.bnd_ax.flat:
            ax.cla(); ax.set_facecolor(LGRAY)
            ax.set_xticks([]); ax.set_yticks([])
        self.bnd_ax[0,0].imshow(np.clip(img,0,1))
        bnd_rgb = np.zeros((*bnd_img.shape,3),dtype=np.float32)
        bnd_rgb[:,:,1] = bnd_img/255.0
        self.bnd_ax[0,1].imshow(bnd_rgb)
        ov_bnd = img.copy(); ov_bnd[bnd_img>0] = [0,1,0.6]
        self.bnd_ax[0,2].imshow(np.clip(ov_bnd,0,1))
        for ax,(t,c) in zip(self.bnd_ax[0],
            [("Original",DGRAY),("Boundaries Only",TEAL),("Overlay",AMBER)]):
            ax.set_title(t,color=c,fontsize=9,fontweight="bold")
        self.bnd_cv.draw()

        # ── SEP TAB ──────────────────────────────────────────────────
        for ax in self.sep_ax.flat:
            ax.cla(); ax.set_facecolor(LGRAY)
            ax.set_xticks([]); ax.set_yticks([])
        self.sep_ax[0,0].imshow(np.clip(
            label2rgb(masks,image=img,bg_label=0,alpha=0.42),0,1))
        self.sep_ax[0,1].imshow(np.clip(
            label2rgb(sep_masks,image=img,bg_label=0,alpha=0.42),0,1))
        self.sep_ax[0,2].imshow(np.clip(img*0.35,0,1))
        
        bnd_mask = find_boundaries(sep_masks, mode='inner')
        bnd_rgba = np.zeros((*sep_masks.shape, 4), dtype=np.float32)
        r,g,b = int(RED[1:3],16)/255, int(RED[3:5],16)/255, int(RED[5:7],16)/255
        bnd_rgba[bnd_mask] = [r, g, b, 0.9]
        self.sep_ax[0,2].imshow(bnd_rgba)
        
        for ax,(t,c) in zip(self.sep_ax[0],[
            (f"Before: {n} masks",AMBER),
            (f"After: {n_sep} cells",TEAL),
            ("Separation lines",RED)]):
            ax.set_title(t,color=c,fontsize=9,fontweight="bold")
        self.sep_stat.config(
            text=f"Before separation: {n}  →  After watershed: {n_sep}  "
                 f"({n_sep-n:+d} cells resolved from overlaps)")
        self.sep_cv.draw()

        # ── PAP TAB ──────────────────────────────────────────────────
        self._update_pap_blood(
            masks,pap_types,PAP_TYPES,img,
            self.pap_ax,self.pap_cv,"Pap Smear")

        # ── BLOOD TAB ────────────────────────────────────────────────
        self._update_pap_blood(
            masks,blood_types,BLOOD_TYPES,img,
            self.bld_ax,self.bld_cv,"Blood Smear")

    def _update_class_tab(self, masks, cell_types, type_dict, img, title):
        self.cls_img_ax.cla(); self.cls_bar_ax.cla(); self.cls_pie_ax.cla()
        for ax in [self.cls_img_ax,self.cls_bar_ax,self.cls_pie_ax]:
            ax.set_facecolor(LGRAY)
        self.cls_img_ax.set_xticks([]); self.cls_img_ax.set_yticks([])

        colored = self._colorize(masks,cell_types,type_dict,img)
        self.cls_img_ax.imshow(np.clip(colored,0,1))
        ct_counts = Counter(cell_types.values())
        self.cls_img_ax.set_title(title,color=TEAL,fontsize=9,fontweight="bold")

        # draw boundary per cell coloured by type instantly
        bnd_mask = find_boundaries(masks, mode='inner')
        bnd_rgb = np.zeros((*masks.shape, 4), dtype=np.float32)
        
        max_id = masks.max()
        cmap = np.zeros((max_id + 1, 4), dtype=np.float32)
        for cid, ct in cell_types.items():
            hex_c = type_dict.get(ct,{}).get("color",TEAL)
            r = int(hex_c[1:3],16)/255
            g = int(hex_c[3:5],16)/255
            b = int(hex_c[5:7],16)/255
            cmap[cid] = [r, g, b, 0.9]
            
        bnd_ids = masks[bnd_mask]
        bnd_rgb[bnd_mask] = cmap[bnd_ids]
        self.cls_img_ax.imshow(bnd_rgb)

        # legend
        patches = [mpatches.Patch(
            color=type_dict.get(t,{}).get("color","#888"),
            label=f"{t} ({ct_counts[t]})")
            for t in ct_counts]
        self.cls_img_ax.legend(handles=patches,loc="lower right",
                                fontsize=7,facecolor=LGRAY,labelcolor=WHITE)

        # bar
        self.cls_bar_ax.tick_params(colors=WHITE,labelsize=8)
        self.cls_bar_ax.set_facecolor(LGRAY)
        if ct_counts:
            bars = self.cls_bar_ax.barh(
                list(ct_counts.keys()),list(ct_counts.values()),
                color=[type_dict.get(t,{}).get("color","#888") for t in ct_counts])
            for bar,v in zip(bars,ct_counts.values()):
                self.cls_bar_ax.text(v+(max(ct_counts.values())*0.02),bar.get_y()+bar.get_height()/2,
                    str(v),va="center",color=WHITE,fontsize=8)
            self.cls_bar_ax.set_yticks([])
            self.cls_bar_ax.set_xlim(0, max(ct_counts.values()) * 1.15)
        self.cls_bar_ax.set_title("Count by Type",color=AMBER,fontsize=9,fontweight="bold")
        self.cls_bar_ax.set_xlabel("Count",color=WHITE,fontsize=8)

        # pie
        if ct_counts:
            cols = [type_dict.get(t,{}).get("color","#888") for t in ct_counts]
            def my_autopct(pct): return f"{pct:.0f}%" if pct > 4 else ""
            self.cls_pie_ax.pie(
                list(ct_counts.values()),
                colors=cols,autopct=my_autopct,startangle=90,
                textprops={"fontsize":7,"color":WHITE})
        self.cls_pie_ax.set_title("Distribution",color=AMBER,fontsize=9,fontweight="bold")

        # legend strip at bottom of tab
        for w in self.cls_leg_frame.winfo_children(): w.destroy()
        for i, (ct,info_d) in enumerate(type_dict.items()):
            tk.Label(self.cls_leg_frame,
                text=f"■ {ct} ({ct_counts.get(ct,0)}) — {info_d['meaning']}",
                font=("Segoe UI",9),fg=info_d["color"],bg=LGRAY
                ).grid(row=i//3, column=i%3, sticky="w", padx=10, pady=2)
        self.cls_cv.draw()

    def _update_pap_blood(self, masks, cell_types, type_dict, img,
                           axes, canvas, label):
        for ax in axes.flat:
            ax.cla(); ax.set_facecolor(LGRAY)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0,0].imshow(np.clip(img,0,1))
        axes[0,0].set_title("Input",color=DGRAY,fontsize=9,fontweight="bold")
        colored = self._colorize(masks,cell_types,type_dict,img)
        axes[0,1].imshow(np.clip(colored,0,1))
        
        bnd_mask = find_boundaries(masks, mode='inner')
        bnd_rgb = np.zeros((*masks.shape, 4), dtype=np.float32)
        max_id = masks.max()
        cmap = np.zeros((max_id + 1, 4), dtype=np.float32)
        for cid, ct in cell_types.items():
            hex_c = type_dict.get(ct,{}).get("color",TEAL)
            r = int(hex_c[1:3],16)/255
            g = int(hex_c[3:5],16)/255
            b = int(hex_c[5:7],16)/255
            cmap[cid] = [r, g, b, 0.9]
        bnd_ids = masks[bnd_mask]
        bnd_rgb[bnd_mask] = cmap[bnd_ids]
        axes[0,1].imshow(bnd_rgb)
        
        ct_counts = Counter(cell_types.values())
        axes[0,1].set_title(f"{label} Classification",
            color=TEAL,fontsize=9,fontweight="bold")
        if ct_counts:
            cols = [type_dict.get(t,{}).get("color","#888") for t in ct_counts]
            def my_autopct(pct): return f"{pct:.0f}%" if pct > 4 else ""
            axes[0,2].pie(list(ct_counts.values()),
                          colors=cols,autopct=my_autopct,startangle=90,
                          textprops={"fontsize":7,"color":WHITE})
        axes[0,2].set_title("Distribution",color=AMBER,fontsize=9,fontweight="bold")
        canvas.draw()

    # ── BOUNDARY HELPER ──────────────────────────────────────────────
    def _get_boundaries(self, masks):
        bnd = find_boundaries(masks, mode='inner')
        return (bnd * 255).astype(np.uint8), {}

    # ── EXPORT ───────────────────────────────────────────────────────
    def _save_vis(self):
        active = [k for k,b in self.tab_btns.items() if b.cget("bg")==TEAL]
        key = active[0] if active else "class"
        figs = {"seg":self.seg_fig,"class":self.cls_fig,
                "local":self.loc_fig,"bnd":self.bnd_fig,
                "sep":self.sep_fig,"pap":self.pap_fig,"blood":self.bld_fig}
        path = filedialog.asksaveasfilename(
            defaultextension=".png",initialfile=f"cell_{key}.png")
        if path:
            figs[key].savefig(path,dpi=180,bbox_inches="tight",facecolor=NAVY)
            self._log(f"Saved: {os.path.basename(path)}")

    def _export_csv(self):
        if self.masks is None: return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",initialfile="cell_data.csv")
        if not path: return
        with open(path,"w",newline="") as f:
            w = csv.writer(f)
            w.writerow(["ID","Cx","Cy","Area","Perimeter",
                        "Eccentricity","Solidity","Tissue_Type"])
            for p in regionprops(self.masks):
                cy,cx = p.centroid
                ct = self.cell_types.get(p.label,"Unknown")
                w.writerow([p.label,f"{cx:.1f}",f"{cy:.1f}",
                             p.area,f"{p.perimeter:.1f}",
                             f"{p.eccentricity:.3f}",f"{p.solidity:.3f}",ct])
        self._log(f"CSV saved")

    def _export_bnd(self):
        if self.masks is None: return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",initialfile="boundaries.csv")
        if not path: return
        self._log("Calculating boundaries for CSV (this may take a moment)...")
        self.root.update_idletasks()
        
        with open(path,"w",newline="") as f:
            w = csv.writer(f); w.writerow(["CellID","X","Y"])
            for cid in np.unique(self.masks):
                if cid==0: continue
                cs = find_contours((self.masks==cid).astype(np.uint8),0.5)
                for c in cs:
                    for pt in c:
                        w.writerow([cid,f"{pt[1]:.1f}",f"{pt[0]:.1f}"])
        self._log("Boundaries exported.")

    # ── HELPERS ──────────────────────────────────────────────────────
    def _refresh_info(self,*_):
        infos = {
            "tissue":
                "H&E Tissue Mode — uses colour deconvolution:\n"
                "Haematoxylin channel (blue) = nuclei\n"
                "Eosin channel (pink) = cytoplasm\n"
                "→ Neoplastic / Inflammatory / Epithelial / Connective / Dead",
            "blood":
                "Blood Smear Mode:\n"
                "→ Neutrophil / Lymphocyte / Monocyte / Eosinophil\n"
                "→ RBC (red cells) / Platelet",
            "pap":
                "Pap Smear Mode — Bethesda system:\n"
                "Normal: Superficial, Intermediate, Parabasal\n"
                "Abnormal: Koilocyte (HPV), Dyskeratotic, Metaplastic",
        }
        self.mode_info.config(text=infos.get(self.analysis_var.get(),""))

    def _log(self, msg):
        self.root.after(0,self._log_main,msg)

    def _log_main(self, msg):
        self.logbox.config(state="normal")
        self.logbox.insert("end",f"▸ {msg}\n")
        self.logbox.see("end")
        self.logbox.config(state="disabled")

    def _status(self, msg):
        self.root.after(0,self.status_lbl.config,{"text":msg})

    def _start_prog(self):
        self.root.after(0,self.prog.start,12)

    def _stop_prog(self):
        self.root.after(0,self.prog.stop)


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    try: style.theme_use("clam")
    except: pass
    style.configure("TProgressbar",troughcolor=LGRAY,
                     background=TEAL,thickness=10)
    app = CellAnalysisGUI(root)
    root.mainloop()
