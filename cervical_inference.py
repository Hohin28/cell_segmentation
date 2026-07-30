import os
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
import numpy as np

class CervicalClassifier:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["ASCUS", "HSIL", "LSIL", "NILM"]
        # ── NOTE: When CCHRC/Bialystok data is added, expand to:
        # self.class_names = ["ASC-H", "ASCUS", "ENDO", "HSIL", "INFL", "LSIL", "NILM", "SCC"]

        # ── Resolve absolute path to cervical_model.pth ───────────────
        # Always look next to THIS script file, regardless of working directory.
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "cervical_model.pth"
        else:
            model_path = Path(model_path).resolve()

        # Initialize EfficientNet-B0 backbone
        self.model = models.efficientnet_b0(weights=None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, len(self.class_names))

        if not model_path.exists():
            print(f"[CervicalAI] cervical_model.pth not found at: {model_path}")
            print(f"[CervicalAI] Run cervical_classifier_train.py first.")
            self.model = None
            return

        try:
            state = torch.load(str(model_path), map_location=self.device, weights_only=False)
            self.model.load_state_dict(state)
            print(f"[CervicalAI] ✅ Loaded weights from: {model_path}")
        except Exception as e:
            print(f"[CervicalAI] ❌ Failed to load weights: {e}")
            self.model = None
            return
            
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.max_batch = 64   # limit VRAM per inference call

    def predict_batch(self, crop_arrays):
        """
        Takes a list of numpy arrays (cropped cells from Cellpose bboxes),
        runs them through EfficientNet-B0, returns list of (class_name, confidence%).
        Safe for grayscale, RGBA, or float32 inputs from the GUI.
        """
        if self.model is None or not crop_arrays:
            return [("–", 0.0)] * len(crop_arrays)

        tensors = []
        for arr in crop_arrays:
            # ── Safety: ensure uint8 RGB ──────────────────────────────
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            if arr.ndim == 2:                          # grayscale
                arr = np.stack([arr]*3, axis=-1)
            elif arr.shape[2] == 4:                    # RGBA
                arr = arr[:, :, :3]
            img = Image.fromarray(arr, mode="RGB")
            tensors.append(self.transform(img))

        # ── Run in mini-batches to guard against VRAM overflow ────────
        all_results = []
        for i in range(0, len(tensors), self.max_batch):
            batch = torch.stack(tensors[i:i + self.max_batch]).to(self.device)
            with torch.no_grad():
                outputs = self.model(batch)
                probs   = torch.nn.functional.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, 1)
            for j in range(len(preds)):
                cls_name = self.class_names[preds[j].item()]
                conf     = confs[j].item() * 100.0
                all_results.append((cls_name, conf))

        return all_results
