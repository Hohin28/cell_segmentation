"""
╔══════════════════════════════════════════════════════════════════╗
║     CELLPOSE-SAM vs CYTO3  —  COMPARISON GUI                   ║
║     Based on: "Cellpose-SAM: Superhuman Generalization          ║
║               for Cellular Segmentation" (2025)                 ║
║                                                                  ║
║  HOW TO RUN (Windows, in your cellpose conda env):              ║
║    conda activate cellpose                                       ║
║    python cellpose_gui.py                                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── standard library ────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import warnings
warnings.filterwarnings("ignore")

# ── scientific / image ───────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from skimage.color import label2rgb
from skimage.measure import regionprops
import tifffile
from PIL import Image, ImageTk

# ── cellpose ─────────────────────────────────────────────────────
import torch
from cellpose import models

# ════════════════════════════════════════════════════════════════
# COLOURS  (matching Cellpose-SAM paper style)
# ════════════════════════════════════════════════════════════════
NAVY   = "#0D1B2A"
TEAL   = "#00A896"
TEAL2  = "#02C39A"
AMBER  = "#FFB74D"
WHITE  = "#FFFFFF"
LGRAY  = "#1E2D3D"
DGRAY  = "#8EACC8"
RED    = "#E24B4A"
GREEN  = "#4CAF50"
PANEL  = "#0F2132"

# ════════════════════════════════════════════════════════════════
# HELPER — metrics (exact paper formulas)
# ════════════════════════════════════════════════════════════════
def compute_metrics(pred, gt, iou_thr=0.5):
    """
    AP  = TP / (TP + FP + FN)
    Err = (FP + FN) / (TP + FN)
    Same formulas as Cellpose-SAM paper Figure 2d-e
    """
    if gt is None:
        return None, None, 0, 0, 0
    pred_ids = np.unique(pred[pred > 0])
    true_ids = np.unique(gt[gt > 0])
    if len(true_ids) == 0:
        return 0.0, 1.0, 0, len(pred_ids), 0
    tp = 0
    matched = set()
    for pid in pred_ids:
        pm = (pred == pid)
        best, btid = 0, -1
        for tid in true_ids:
            if tid in matched:
                continue
            tm = (gt == tid)
            inter = np.logical_and(pm, tm).sum()
            union = np.logical_or(pm, tm).sum()
            iou = inter / (union + 1e-8)
            if iou > best:
                best, btid = iou, tid
        if best >= iou_thr:
            tp += 1
            matched.add(btid)
    fp = len(pred_ids) - tp
    fn = len(true_ids) - tp
    ap  = tp / (tp + fp + fn + 1e-8)
    err = (fp + fn) / (tp + fn + 1e-8)
    return ap, err, tp, fp, fn


# ════════════════════════════════════════════════════════════════
# MAIN GUI CLASS
# ════════════════════════════════════════════════════════════════
class CellposeGUI:

    def __init__(self, root):
        self.root = root
        self.root.title(
            "Cellpose-SAM  vs  cyto3  —  Comparison Tool  "
            "|  Based on Pachitariu et al. 2025"
        )
        self.root.configure(bg=NAVY)
        self.root.state("zoomed")          # start maximised on Windows

        # ── state ────────────────────────────────────────────────
        self.img_path   = None
        self.img_array  = None            # loaded image (H,W,3) or (H,W)
        self.gt_masks   = None            # optional ground-truth
        self.masks_old  = None            # cyto3 result
        self.masks_new  = None            # cpsam result
        self.model_cyto3 = None
        self.model_cpsam = None

        self._build_ui()
        self._log("GUI ready.  Load an image to begin.")
        self._log(f"GPU available: {torch.cuda.is_available()}")

    # ── UI BUILDER ───────────────────────────────────────────────
    def _build_ui(self):
        # ── top banner ──────────────────────────────────────────
        banner = tk.Frame(self.root, bg=NAVY, height=56)
        banner.pack(fill="x", side="top")
        tk.Label(
            banner,
            text="Cellpose-SAM  vs  cyto3  —  Cell Segmentation Comparison",
            font=("Segoe UI", 16, "bold"), fg=TEAL, bg=NAVY
        ).pack(side="left", padx=18, pady=10)
        tk.Label(
            banner,
            text="Pachitariu et al., bioRxiv 2025",
            font=("Segoe UI", 10), fg=DGRAY, bg=NAVY
        ).pack(side="right", padx=18)

        # ── main body = left panel + right canvas ────────────────
        body = tk.Frame(self.root, bg=NAVY)
        body.pack(fill="both", expand=True)

        self._build_left_panel(body)
        self._build_right_canvas(body)

    # ── LEFT CONTROL PANEL ───────────────────────────────────────
    def _build_left_panel(self, parent):
        lf = tk.Frame(parent, bg=PANEL, width=300)
        lf.pack(side="left", fill="y", padx=(10, 5), pady=10)
        lf.pack_propagate(False)

        def section(text):
            tk.Label(lf, text=text, font=("Segoe UI", 10, "bold"),
                     fg=TEAL, bg=PANEL).pack(anchor="w", padx=14,
                                              pady=(14, 2))
            tk.Frame(lf, bg=TEAL, height=1).pack(fill="x",
                                                   padx=14, pady=(0, 8))

        def btn(parent, text, cmd, color=TEAL, fg=NAVY):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=color, fg=fg, font=("Segoe UI", 9, "bold"),
                          relief="flat", cursor="hand2", padx=6, pady=5)
            b.pack(fill="x", padx=14, pady=3)
            return b

        # ── 1. Load image ─────────────────────────────────────────
        section("1.  Load Image")
        btn(lf, "📂  Browse Image…", self._load_image)
        self.lbl_file = tk.Label(lf, text="No file loaded",
                                  font=("Segoe UI", 8), fg=DGRAY,
                                  bg=PANEL, wraplength=270)
        self.lbl_file.pack(anchor="w", padx=14, pady=(0, 4))
        btn(lf, "📂  Load Ground Truth (optional)",
            self._load_gt, color=LGRAY, fg=WHITE)
        self.lbl_gt = tk.Label(lf, text="No GT loaded",
                                font=("Segoe UI", 8), fg=DGRAY,
                                bg=PANEL, wraplength=270)
        self.lbl_gt.pack(anchor="w", padx=14, pady=(0, 4))

        # ── 2. Model settings ─────────────────────────────────────
        section("2.  Model Settings")

        tk.Label(lf, text="Cell diameter (px):",
                 font=("Segoe UI", 9), fg=WHITE, bg=PANEL
                 ).pack(anchor="w", padx=14)
        self.diam_var = tk.StringVar(value="30")
        tk.Entry(lf, textvariable=self.diam_var,
                 font=("Segoe UI", 9), width=10,
                 bg=LGRAY, fg=WHITE, insertbackground=WHITE
                 ).pack(anchor="w", padx=14, pady=(2, 8))

        tk.Label(lf, text="Flow threshold:",
                 font=("Segoe UI", 9), fg=WHITE, bg=PANEL
                 ).pack(anchor="w", padx=14)
        self.flow_var = tk.StringVar(value="0.4")
        tk.Entry(lf, textvariable=self.flow_var,
                 font=("Segoe UI", 9), width=10,
                 bg=LGRAY, fg=WHITE, insertbackground=WHITE
                 ).pack(anchor="w", padx=14, pady=(2, 8))

        tk.Label(lf, text="Cell prob threshold:",
                 font=("Segoe UI", 9), fg=WHITE, bg=PANEL
                 ).pack(anchor="w", padx=14)
        self.prob_var = tk.StringVar(value="0.0")
        tk.Entry(lf, textvariable=self.prob_var,
                 font=("Segoe UI", 9), width=10,
                 bg=LGRAY, fg=WHITE, insertbackground=WHITE
                 ).pack(anchor="w", padx=14, pady=(2, 8))

        # ── 3. Run ────────────────────────────────────────────────
        section("3.  Run Segmentation")

        self.run_both_btn = btn(
            lf, "▶  Run BOTH Models", self._run_both,
            color=TEAL, fg=NAVY)
        btn(lf, "▶  Run cyto3 only", self._run_cyto3_only,
            color=AMBER, fg=NAVY)
        btn(lf, "▶  Run CPSAM only", self._run_cpsam_only,
            color="#7C6AF5", fg=WHITE)

        # progress bar
        self.prog = ttk.Progressbar(lf, mode="indeterminate",
                                     length=260)
        self.prog.pack(padx=14, pady=6)

        self.status_lbl = tk.Label(lf, text="Idle",
                                    font=("Segoe UI", 9),
                                    fg=TEAL2, bg=PANEL)
        self.status_lbl.pack(anchor="w", padx=14)

        # ── 4. Results summary ────────────────────────────────────
        section("4.  Results Summary")
        self.result_text = tk.Text(
            lf, height=12, bg=LGRAY, fg=WHITE,
            font=("Consolas", 8), relief="flat",
            state="disabled", wrap="word"
        )
        self.result_text.pack(fill="x", padx=14, pady=4)

        # ── 5. Save ───────────────────────────────────────────────
        section("5.  Export")
        btn(lf, "💾  Save Comparison Image",
            self._save_figure, color=GREEN, fg=WHITE)
        btn(lf, "💾  Save Masks (NPY)",
            self._save_masks, color=LGRAY, fg=WHITE)

        # ── log ───────────────────────────────────────────────────
        section("Log")
        self.log_text = tk.Text(
            lf, height=8, bg=LGRAY, fg=DGRAY,
            font=("Consolas", 7), relief="flat",
            state="disabled", wrap="word"
        )
        self.log_text.pack(fill="x", padx=14, pady=(0, 10))

    # ── RIGHT CANVAS (matplotlib figures) ───────────────────────
    def _build_right_canvas(self, parent):
        rf = tk.Frame(parent, bg=NAVY)
        rf.pack(side="left", fill="both", expand=True,
                padx=(5, 10), pady=10)

        # title
        tk.Label(
            rf,
            text="Segmentation Results  —  Input  |  cyto3  |  Cellpose-SAM",
            font=("Segoe UI", 11, "bold"), fg=WHITE, bg=NAVY
        ).pack(anchor="w", pady=(0, 4))

        # main figure: 2 rows × 3 cols
        self.fig = Figure(figsize=(12, 8), facecolor=NAVY)
        self.fig.subplots_adjust(hspace=0.35, wspace=0.05,
                                  left=0.03, right=0.97,
                                  top=0.93, bottom=0.04)
        self.axes = self.fig.subplots(2, 3)

        for ax in self.axes.flat:
            ax.set_facecolor(LGRAY)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#1E3A4A")

        # column headers
        headers  = ["Input Image", "cyto3  (OLD model)",
                    "Cellpose-SAM  (NEW model)"]
        h_colors = [DGRAY, AMBER, TEAL]
        for col, (h, c) in enumerate(zip(headers, h_colors)):
            self.axes[0, col].set_title(h, color=c,
                                         fontsize=10, fontweight="bold",
                                         pad=6)

        # row labels
        self.axes[0, 0].set_ylabel("Full image",
                                    color=WHITE, fontsize=9)
        self.axes[1, 0].set_ylabel("Zoomed (centre)",
                                    color=WHITE, fontsize=9)

        self.canvas = FigureCanvasTkAgg(self.fig, master=rf)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        # metrics bar chart (separate small figure)
        tk.Label(
            rf,
            text="Paper Metrics  —  AP @ IoU=0.5  |  Error Rate  |  Speed",
            font=("Segoe UI", 10, "bold"), fg=WHITE, bg=NAVY
        ).pack(anchor="w", pady=(8, 2))

        self.fig2 = Figure(figsize=(12, 2.6), facecolor=NAVY)
        self.fig2.subplots_adjust(left=0.06, right=0.98,
                                   top=0.82, bottom=0.22,
                                   wspace=0.3)
        self.axes2 = self.fig2.subplots(1, 3)
        for ax in self.axes2:
            ax.set_facecolor(LGRAY)
            for sp in ax.spines.values():
                sp.set_edgecolor("#1E3A4A")
            ax.tick_params(colors=WHITE, labelsize=8)
            ax.yaxis.label.set_color(WHITE)
            ax.title.set_color(WHITE)

        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=rf)
        self.canvas2.get_tk_widget().pack(fill="x", pady=(0, 4))
        self.canvas2.draw()

    # ── FILE LOADING ─────────────────────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select microscopy image",
            filetypes=[
                ("Image files",
                 "*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.npy"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return
        self.img_path = path
        try:
            if path.endswith(".npy"):
                arr = np.load(path)
            else:
                arr = tifffile.imread(path)
            # normalise to 0-1 float
            arr = arr.astype(np.float32)
            if arr.max() > 1:
                arr = arr / arr.max()
            # ensure 3-channel
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            elif arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.moveaxis(arr, 0, -1)
                if arr.shape[2] == 1:
                    arr = np.concatenate([arr]*3, axis=-1)
            self.img_array = arr
            short = path.split("/")[-1].split("\\")[-1]
            self.lbl_file.config(text=short)
            self._log(f"Loaded: {short}  "
                      f"({arr.shape[0]}×{arr.shape[1]}px)")
            self._draw_input_only()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _load_gt(self):
        path = filedialog.askopenfilename(
            title="Select ground-truth mask (NPY or TIFF)",
            filetypes=[("Mask files", "*.npy *.tif *.tiff"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        try:
            if path.endswith(".npy"):
                self.gt_masks = np.load(path,
                                         allow_pickle=True)
                if self.gt_masks.dtype == object:
                    self.gt_masks = self.gt_masks.item()
                    if isinstance(self.gt_masks, dict):
                        self.gt_masks = self.gt_masks.get(
                            "masks", None)
            else:
                self.gt_masks = tifffile.imread(path
                                                 ).astype(np.int32)
            short = path.split("/")[-1].split("\\")[-1]
            n = int(self.gt_masks.max()) if \
                self.gt_masks is not None else 0
            self.lbl_gt.config(text=f"{short}  ({n} GT cells)")
            self._log(f"GT loaded: {short}  ({n} cells)")
        except Exception as e:
            messagebox.showerror("GT load error", str(e))

    # ── MODEL LOADING (lazy) ─────────────────────────────────────
    def _ensure_models(self, need_cyto3=True, need_cpsam=True):
        dev = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if need_cyto3 and self.model_cyto3 is None:
            self._status("Loading cyto3 weights…")
            self.model_cyto3 = models.CellposeModel(
                model_type="cyto3", device=dev)
            self._log("cyto3 model loaded.")
        if need_cpsam and self.model_cpsam is None:
            self._status("Loading Cellpose-SAM weights…")
            self.model_cpsam = models.CellposeModel(
                model_type="cpsam", device=dev)
            self._log("Cellpose-SAM model loaded.")

    # ── SEGMENTATION RUNNER ──────────────────────────────────────
    def _segment(self, model, img_arr):
        gray = (0.299 * img_arr[:, :, 0] +
                0.587 * img_arr[:, :, 1] +
                0.114 * img_arr[:, :, 2])
        gray_u8 = (gray * 255).astype(np.uint8)
        try:
            diam  = float(self.diam_var.get())
        except Exception:
            diam = 30.0
        try:
            flow  = float(self.flow_var.get())
        except Exception:
            flow = 0.4
        try:
            prob  = float(self.prob_var.get())
        except Exception:
            prob = 0.0

        t0 = time.time()
        masks, _, _ = model.eval(
            gray_u8,
            diameter=diam,
            channels=[0, 0],
            flow_threshold=flow,
            cellprob_threshold=prob,
        )
        runtime = time.time() - t0
        return masks, runtime

    # ── RUN BUTTONS ──────────────────────────────────────────────
    def _run_both(self):
        if self.img_array is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        threading.Thread(target=self._run_both_thread,
                         daemon=True).start()

    def _run_both_thread(self):
        self._start_progress()
        try:
            self._ensure_models(True, True)
            self._status("Running cyto3…")
            self.masks_old, t_old = self._segment(
                self.model_cyto3, self.img_array)
            n_old = len(np.unique(self.masks_old)) - 1
            self._log(f"cyto3 → {n_old} cells  ({t_old:.2f}s)")

            self._status("Running Cellpose-SAM…")
            self.masks_new, t_new = self._segment(
                self.model_cpsam, self.img_array)
            n_new = len(np.unique(self.masks_new)) - 1
            self._log(f"CPSAM  → {n_new} cells  ({t_new:.2f}s)")

            # metrics
            ap_old, err_old, tp_o, fp_o, fn_o = \
                compute_metrics(self.masks_old, self.gt_masks)
            ap_new, err_new, tp_n, fp_n, fn_n = \
                compute_metrics(self.masks_new, self.gt_masks)

            self.root.after(0, self._update_ui,
                             n_old, t_old, n_new, t_new,
                             ap_old, err_old, tp_o, fp_o, fn_o,
                             ap_new, err_new, tp_n, fp_n, fn_n)
        except Exception as e:
            self._log(f"ERROR: {e}")
            self.root.after(0, messagebox.showerror,
                             "Run error", str(e))
        finally:
            self._stop_progress()
            self._status("Done.")

    def _run_cyto3_only(self):
        if self.img_array is None:
            messagebox.showwarning("No image",
                                   "Please load an image first.")
            return
        threading.Thread(target=self._run_single_thread,
                         target_args=("cyto3",),
                         daemon=True).start()

    def _run_cpsam_only(self):
        if self.img_array is None:
            messagebox.showwarning("No image",
                                   "Please load an image first.")
            return
        threading.Thread(target=self._run_single_thread,
                         args=("cpsam",),
                         daemon=True).start()

    def _run_single_thread(self, which):
        self._start_progress()
        try:
            if which == "cyto3":
                self._ensure_models(True, False)
                self._status("Running cyto3…")
                self.masks_old, t = self._segment(
                    self.model_cyto3, self.img_array)
                n = len(np.unique(self.masks_old)) - 1
                self._log(f"cyto3 → {n} cells  ({t:.2f}s)")
            else:
                self._ensure_models(False, True)
                self._status("Running Cellpose-SAM…")
                self.masks_new, t = self._segment(
                    self.model_cpsam, self.img_array)
                n = len(np.unique(self.masks_new)) - 1
                self._log(f"CPSAM → {n} cells  ({t:.2f}s)")

            self.root.after(0, self._redraw_main)
        except Exception as e:
            self._log(f"ERROR: {e}")
        finally:
            self._stop_progress()
            self._status("Done.")

    # ── UI UPDATE (called on main thread) ────────────────────────
    def _update_ui(self, n_old, t_old, n_new, t_new,
                   ap_old, err_old, tp_o, fp_o, fn_o,
                   ap_new, err_new, tp_n, fp_n, fn_n):
        self._redraw_main()
        self._redraw_metrics(
            n_old, t_old, n_new, t_new,
            ap_old, err_old, tp_o, fp_o, fn_o,
            ap_new, err_new, tp_n, fp_n, fn_n)
        self._fill_results_box(
            n_old, t_old, n_new, t_new,
            ap_old, err_old, tp_o, fp_o, fn_o,
            ap_new, err_new, tp_n, fp_n, fn_n)

    # ── DRAW INPUT ONLY ──────────────────────────────────────────
    def _draw_input_only(self):
        img = self.img_array
        for r in range(2):
            for c in range(3):
                self.axes[r, c].cla()
                self.axes[r, c].set_xticks([])
                self.axes[r, c].set_yticks([])
                self.axes[r, c].set_facecolor(LGRAY)

        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        zh, zw = h // 4, w // 4

        # top row: full image in col 0
        self.axes[0, 0].imshow(np.clip(img, 0, 1))
        self.axes[0, 0].set_title("Input Image",
                                   color=DGRAY,
                                   fontsize=10, fontweight="bold")
        # bottom row: zoom in col 0
        crop = img[max(0,cy-zh):cy+zh, max(0,cx-zw):cx+zw]
        self.axes[1, 0].imshow(np.clip(crop, 0, 1))

        # placeholders
        for c in (1, 2):
            for r in (0, 1):
                self.axes[r, c].text(
                    0.5, 0.5, "Run model to see result",
                    ha="center", va="center",
                    color=DGRAY, fontsize=9,
                    transform=self.axes[r, c].transAxes)

        self.axes[0, 0].set_ylabel("Full image",
                                    color=WHITE, fontsize=9)
        self.axes[1, 0].set_ylabel("Zoomed (centre)",
                                    color=WHITE, fontsize=9)
        self.canvas.draw()

    # ── DRAW MAIN COMPARISON ─────────────────────────────────────
    def _redraw_main(self):
        img = self.img_array
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        zh, zw = h // 4, w // 4

        masks = [None, self.masks_old, self.masks_new]
        titles = ["Input Image",
                  "cyto3  (OLD)",
                  "Cellpose-SAM  (NEW)"]
        t_colors = [DGRAY, AMBER, TEAL]

        def n_cells(m):
            return len(np.unique(m)) - 1 if m is not None else 0

        for r in range(2):
            for c in range(3):
                ax = self.axes[r, c]
                ax.cla()
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_facecolor(LGRAY)

                m = masks[c]
                if m is None:
                    if c == 0:
                        show = np.clip(img, 0, 1)
                        if r == 1:
                            show = show[max(0, cy-zh):cy+zh,
                                        max(0, cx-zw):cx+zw]
                        ax.imshow(show)
                    else:
                        ax.text(0.5, 0.5,
                                "Run model\nto see result",
                                ha="center", va="center",
                                color=DGRAY, fontsize=9,
                                transform=ax.transAxes)
                else:
                    if r == 0:
                        overlay = label2rgb(
                            m, image=img, bg_label=0,
                            alpha=0.40)
                        ax.imshow(np.clip(overlay, 0, 1))
                    else:
                        m_crop = m[max(0, cy-zh):cy+zh,
                                   max(0, cx-zw):cx+zw]
                        i_crop = img[max(0, cy-zh):cy+zh,
                                     max(0, cx-zw):cx+zw]
                        overlay = label2rgb(
                            m_crop, image=i_crop,
                            bg_label=0, alpha=0.45)
                        ax.imshow(np.clip(overlay, 0, 1))

                if r == 0:
                    n = n_cells(m)
                    sub = (f"  {n} cells detected"
                           if m is not None else "")
                    ax.set_title(
                        f"{titles[c]}{sub}",
                        color=t_colors[c],
                        fontsize=9, fontweight="bold", pad=5)

        self.axes[0, 0].set_ylabel("Full image",
                                    color=WHITE, fontsize=9)
        self.axes[1, 0].set_ylabel("Zoomed (centre)",
                                    color=WHITE, fontsize=9)

        # Draw rectangle on full images showing zoom area
        for c in range(3):
            if masks[c] is not None or c == 0:
                rect = plt.Rectangle(
                    (max(0, cx-zw), max(0, cy-zh)),
                    min(2*zw, w), min(2*zh, h),
                    linewidth=1.5,
                    edgecolor=TEAL, facecolor="none",
                    linestyle="--")
                self.axes[0, c].add_patch(rect)

        self.canvas.draw()

    # ── DRAW METRICS CHART ───────────────────────────────────────
    def _redraw_metrics(self,
                         n_old, t_old, n_new, t_new,
                         ap_old, err_old, tp_o, fp_o, fn_o,
                         ap_new, err_new, tp_n, fp_n, fn_n):
        for ax in self.axes2:
            ax.cla()
            ax.set_facecolor(LGRAY)
            ax.tick_params(colors=WHITE, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#1E3A4A")

        labels = ["cyto3", "Cellpose-SAM"]
        x = np.array([0, 1])
        w = 0.5

        # ── Chart 1: cells detected ──────────────────────────────
        bars = self.axes2[0].bar(
            x, [n_old, n_new], w,
            color=[AMBER, TEAL], alpha=0.88)
        self.axes2[0].set_xticks(x)
        self.axes2[0].set_xticklabels(labels,
                                       color=WHITE, fontsize=8)
        self.axes2[0].set_title("Cells Detected",
                                 color=WHITE, fontsize=9,
                                 fontweight="bold")
        self.axes2[0].set_ylabel("Count",
                                  color=WHITE, fontsize=8)
        for bar, v in zip(bars, [n_old, n_new]):
            self.axes2[0].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                str(v), ha="center",
                color=WHITE, fontsize=9, fontweight="bold")

        # ── Chart 2: AP score ────────────────────────────────────
        if ap_old is not None and ap_new is not None:
            bars2 = self.axes2[1].bar(
                x, [ap_old, ap_new], w,
                color=[AMBER, TEAL], alpha=0.88)
            self.axes2[1].axhline(
                0.257, color=WHITE, ls="--", lw=1,
                label="Inter-human (0.257)")
            self.axes2[1].axhline(
                0.163, color=TEAL2, ls=":", lw=1,
                label="Paper CPSAM (0.163 err)")
            self.axes2[1].set_ylim(0, 1.15)
            self.axes2[1].legend(fontsize=6.5,
                                  facecolor=LGRAY,
                                  labelcolor=WHITE,
                                  loc="upper left")
            for bar, v in zip(bars2,
                               [ap_old, ap_new]):
                self.axes2[1].text(
                    bar.get_x() + bar.get_width()/2,
                    v + 0.02, f"{v:.3f}",
                    ha="center", color=WHITE,
                    fontsize=8, fontweight="bold")
        else:
            self.axes2[1].text(
                0.5, 0.5,
                "Load GT mask\nfor AP metrics",
                ha="center", va="center",
                color=DGRAY, fontsize=8,
                transform=self.axes2[1].transAxes)

        self.axes2[1].set_xticks(x)
        self.axes2[1].set_xticklabels(labels,
                                       color=WHITE, fontsize=8)
        self.axes2[1].set_title(
            "AP @ IoU=0.5  (Higher = Better)",
            color=WHITE, fontsize=9, fontweight="bold")

        # ── Chart 3: Speed ───────────────────────────────────────
        bars3 = self.axes2[2].bar(
            x, [t_old, t_new], w,
            color=[AMBER, TEAL], alpha=0.88)
        self.axes2[2].set_xticks(x)
        self.axes2[2].set_xticklabels(labels,
                                       color=WHITE, fontsize=8)
        self.axes2[2].set_title(
            "Inference Time  (Lower = Better)",
            color=WHITE, fontsize=9, fontweight="bold")
        self.axes2[2].set_ylabel("Seconds",
                                  color=WHITE, fontsize=8)
        for bar, v in zip(bars3, [t_old, t_new]):
            self.axes2[2].text(
                bar.get_x() + bar.get_width()/2,
                v + 0.01,
                f"{v:.2f}s",
                ha="center", color=WHITE,
                fontsize=8, fontweight="bold")

        self.canvas2.draw()

    # ── RESULTS TEXT BOX ─────────────────────────────────────────
    def _fill_results_box(self,
                           n_old, t_old, n_new, t_new,
                           ap_old, err_old, tp_o, fp_o, fn_o,
                           ap_new, err_new, tp_n, fp_n, fn_n):
        has_gt = ap_old is not None

        lines = [
            "══ cyto3 ══",
            f"  Cells detected : {n_old}",
            f"  Time           : {t_old:.2f}s",
        ]
        if has_gt:
            lines += [
                f"  AP @ 0.5 IoU   : {ap_old:.3f}",
                f"  Error rate     : {err_old:.3f}",
                f"  TP={tp_o}  FP={fp_o}  FN={fn_o}",
            ]
        lines += [
            "",
            "══ Cellpose-SAM ══",
            f"  Cells detected : {n_new}",
            f"  Time           : {t_new:.2f}s",
        ]
        if has_gt:
            lines += [
                f"  AP @ 0.5 IoU   : {ap_new:.3f}",
                f"  Error rate     : {err_new:.3f}",
                f"  TP={tp_n}  FP={fp_n}  FN={fn_n}",
            ]
        lines += [
            "",
            "── Paper benchmarks ──",
            "  cyto3 error    : 0.292",
            "  CPSAM error    : 0.163",
            "  Inter-human    : 0.257",
            "  Consensus est. : 0.128",
        ]
        if has_gt:
            winner = ("CPSAM ✅" if ap_new >= ap_old
                      else "cyto3 ✅")
            lines += ["", f"  WINNER: {winner}"]

        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "\n".join(lines))
        self.result_text.config(state="disabled")

    # ── SAVE FIGURE ──────────────────────────────────────────────
    def _save_figure(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"),
                       ("PDF", "*.pdf")],
            initialfile="cellpose_comparison.png"
        )
        if path:
            try:
                self.fig.savefig(path, dpi=180,
                                  bbox_inches="tight",
                                  facecolor=NAVY)
                self._log(f"Saved: {path.split('/')[-1]}")
                messagebox.showinfo("Saved", f"Saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Save error", str(e))

    def _save_masks(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy", "*.npy")],
            initialfile="masks_result.npy"
        )
        if path:
            try:
                data = {
                    "masks_cyto3": self.masks_old,
                    "masks_cpsam": self.masks_new,
                    "image_path": self.img_path,
                }
                np.save(path, data)
                self._log(f"Masks saved: "
                           f"{path.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Save error", str(e))

    # ── HELPERS ──────────────────────────────────────────────────
    def _log(self, msg):
        self.root.after(0, self._log_main, msg)

    def _log_main(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"▸ {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _status(self, msg):
        self.root.after(0, self.status_lbl.config, {"text": msg})

    def _start_progress(self):
        self.root.after(0, self.prog.start, 12)

    def _stop_progress(self):
        self.root.after(0, self.prog.stop)


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()

    # ── style tweaks ─────────────────────────────────────────────
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TProgressbar",
                     troughcolor=LGRAY,
                     background=TEAL,
                     thickness=10)

    app = CellposeGUI(root)
    root.mainloop()
