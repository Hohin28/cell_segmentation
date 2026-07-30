import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy

# ==============================================================================
# CERVICAL CELL EFFICIENTNET-B0 TRAINING PIPELINE
# ==============================================================================

# The 4 mandatory Bethesda classes covered by SIPaKMeD.
# When CCHRC/Bialystok data is added, expand to 8 classes.
REQUIRED_CLASSES = ["ASCUS", "HSIL", "LSIL", "NILM"]

# ── EfficientNet-B0 Training Thresholds (Research-Backed) ────────────────────
# These are the MINIMUM images per class recommended for stable transfer learning.
# Source: Based on standard transfer learning literature for medical imaging.
#   < 50   images → CRITICAL: Model will overfit immediately. Do not train.
#   50-100 images → WARNING: Very high overfitting risk. Use heavy augmentation.
#   100-300 images → ACCEPTABLE: Workable with Transfer Learning + Augmentation.
#   300-500 images → GOOD: Stable training expected.
#   500+   images → IDEAL: Reliable generalization expected.
MINIMUM_IMAGES_CRITICAL = 50    # Hard stop threshold
MINIMUM_IMAGES_WARNING  = 100   # Print warning but allow training
MINIMUM_IMAGES_GOOD     = 300   # Ideal target per class
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# =============================================================================

def verify_dataset(data_dir="training_crops"):
    """
    Performs a comprehensive verification of the training dataset.
    Prints a detailed report and returns False if any class is empty or missing,
    which will abort the training pipeline before any GPU resources are used.
    """
    print("=" * 65)
    print("  DATASET VERIFICATION REPORT")
    print("=" * 65)

    # ── 1. Check if the root directory exists ─────────────────────────────
    if not os.path.exists(data_dir):
        print(f"\n[FATAL] Root directory not found: '{os.path.abspath(data_dir)}'")
        print("        Create the folder and populate it with class sub-folders.")
        return False

    # ── 2. Scan all class folders and count images ─────────────────────────
    present_classes = {}
    for entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if entry.is_dir():
            count = sum(
                1 for f in os.scandir(entry.path)
                if f.is_file() and f.name.lower().endswith(IMG_EXTENSIONS)
            )
            present_classes[entry.name] = count

    # ── 3. Identify missing and empty classes ─────────────────────────────
    missing_classes = [c for c in REQUIRED_CLASSES if c not in present_classes]
    empty_classes   = [c for c, n in present_classes.items() if n == 0]
    found_required  = {c: present_classes[c] for c in REQUIRED_CLASSES if c in present_classes}
    extra_classes   = [c for c in present_classes if c not in REQUIRED_CLASSES]

    # ── 4. Per-Class Statistics Table ─────────────────────────────────────
    print(f"\n{'Class':<10} {'Images':>8}  {'Status'}")
    print("-" * 45)

    has_any_critical = False
    for cls in REQUIRED_CLASSES:
        count = present_classes.get(cls, 0)
        if cls in missing_classes:
            status = "❌  MISSING — folder does not exist"
            has_any_critical = True
        elif count == 0:
            status = "❌  EMPTY   — folder exists but has no images"
            has_any_critical = True
        elif count < MINIMUM_IMAGES_CRITICAL:
            status = f"🔴  CRITICAL  — fewer than {MINIMUM_IMAGES_CRITICAL} images"
            has_any_critical = True
        elif count < MINIMUM_IMAGES_WARNING:
            status = f"🟠  WARNING  — fewer than {MINIMUM_IMAGES_WARNING} images"
        elif count < MINIMUM_IMAGES_GOOD:
            status = f"🟡  ACCEPTABLE — fewer than {MINIMUM_IMAGES_GOOD} images"
        else:
            status = "🟢  GOOD"
        print(f"  {cls:<10} {count:>6}    {status}")

    if extra_classes:
        print()
        for cls in extra_classes:
            print(f"  {cls:<10} {present_classes[cls]:>6}    ⚪  EXTRA (not in required classes)")

    # ── 5. Summary Statistics ─────────────────────────────────────────────
    total_images = sum(present_classes.values())
    print("\n" + "=" * 65)
    print(f"  Total Images Found         : {total_images}")
    print(f"  Required Classes           : {len(REQUIRED_CLASSES)}")
    print(f"  Classes Present            : {len(present_classes)}")
    print(f"  Missing Classes            : {len(missing_classes)}"
          + (f"  → {missing_classes}" if missing_classes else ""))
    print(f"  Empty Classes              : {len(empty_classes)}"
          + (f"  → {empty_classes}" if empty_classes else ""))
    print(f"  Extra (Unexpected) Classes : {len(extra_classes)}"
          + (f"  → {extra_classes}" if extra_classes else ""))

    # ── 6. Imbalance Ratio ────────────────────────────────────────────────
    print("\n── Class Imbalance Analysis ──────────────────────────────────")
    counts_list = [present_classes.get(c, 0) for c in REQUIRED_CLASSES]
    valid_counts = [c for c in counts_list if c > 0]
    if valid_counts:
        max_count  = max(valid_counts)
        min_count  = min(valid_counts)
        max_class  = REQUIRED_CLASSES[counts_list.index(max_count)]
        min_class  = REQUIRED_CLASSES[counts_list.index(min_count)]
        ratio      = max_count / min_count if min_count > 0 else float('inf')
        print(f"  Largest class  : {max_class} ({max_count} images)")
        print(f"  Smallest class : {min_class} ({min_count} images)")
        print(f"  Imbalance Ratio: {ratio:.1f}x")
        if ratio > 20:
            print("  ⚠️  SEVERE imbalance. Class-Weighted Loss is mandatory.")
        elif ratio > 5:
            print("  ⚠️  MODERATE imbalance. Class-Weighted Loss is strongly recommended.")
        else:
            print("  ✅  Imbalance is within acceptable range.")
    else:
        print("  No valid data to compute imbalance.")

    # ── 7. Expected Train / Validation / Test Splits ──────────────────────
    train_n = int(total_images * 0.70)
    val_n   = int(total_images * 0.15)
    test_n  = total_images - train_n - val_n
    print("\n── Expected Train / Val / Test Splits (70 / 15 / 15) ──────────")
    print(f"  Train : {train_n} images")
    print(f"  Val   : {val_n} images")
    print(f"  Test  : {test_n} images")

    # ── 8. Minimum Recommended Thresholds Reference ───────────────────────
    print("\n── EfficientNet-B0 Minimum Recommendations per Class ──────────")
    print(f"  🔴  < {MINIMUM_IMAGES_CRITICAL:>4}  : CRITICAL — Do NOT train. Will overfit immediately.")
    print(f"  🟠  < {MINIMUM_IMAGES_WARNING:>4}  : WARNING  — High overfit risk. Augmentation required.")
    print(f"  🟡  < {MINIMUM_IMAGES_GOOD:>4}  : ACCEPTABLE — Transfer learning + augmentation OK.")
    print(f"  🟢  >= {MINIMUM_IMAGES_GOOD:<4} : GOOD — Reliable training expected.")
    print("=" * 65)

    # ── 9. Final Go / No-Go Decision ─────────────────────────────────────
    if has_any_critical:
        print("\n[ABORT] One or more required classes are missing, empty, or critically")
        print("        under-populated. Training has been STOPPED to prevent a broken model.")
        print("        Please populate all 8 class folders before proceeding.")
        return False

    print("\n[OK] All required classes are present. Proceeding to training...\n")
    return True


# =============================================================================

def train_model(data_dir="training_crops", num_epochs=30, batch_size=16, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    class_names  = full_dataset.classes
    print(f"\nClass-to-Index mapping confirmed by PyTorch ImageFolder:")
    for idx, name in enumerate(class_names):
        print(f"  Index {idx} → {name}")

    # 80/20 Train/Val Split (Test set is held out manually by user)
    train_size   = int(0.8 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = data_transforms['val']

    dataloaders  = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2),
        'val':   DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Class Imbalance via Weighted Loss
    class_counts = np.zeros(len(class_names))
    for _, label in train_dataset:
        class_counts[label] += 1
    weights      = 1.0 / (class_counts + 1e-6)
    weights      = weights / np.sum(weights) * len(class_names)
    class_weights = torch.FloatTensor(weights).to(device)
    print("\nComputed Class Weights for CrossEntropyLoss:")
    for name, w in zip(class_names, class_weights.cpu().numpy()):
        print(f"  {name:<10}: {w:.4f}")

    # Transfer Learning Setup (EfficientNet-B0)
    model = models.efficientnet_b0(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    # Training Loop
    best_model_wts  = copy.deepcopy(model.state_dict())
    best_acc        = 0.0
    patience_counter = 0
    max_patience    = 7

    print("\n" + "=" * 65)
    print("  STARTING TRAINING")
    print("=" * 65)

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward(); optimizer.step()
                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            print(f'  {phase.capitalize():<6} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    print(f"  ✅ New best model saved (Val Acc: {best_acc:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{max_patience})")

        if patience_counter >= max_patience:
            print("\nEarly stopping triggered!")
            break

    print(f'\n{"=" * 65}')
    print(f'  Training complete. Best Val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cervical_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"  Model saved to: {save_path}")
    print(f'{"=" * 65}\n')

    evaluate_model(model, dataloaders['val'], class_names, device)


def evaluate_model(model, val_loader, class_names, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_matrix.png')
    plt.savefig(report_path)
    print(f"Confusion matrix saved to: {report_path}")


if __name__ == '__main__':
    if verify_dataset(data_dir="training_crops"):
        train_model(data_dir="training_crops", num_epochs=30, batch_size=16)
