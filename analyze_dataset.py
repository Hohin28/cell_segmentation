import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from skimage.color import rgb2hed
import random

def analyze_dataset(dataset_dir="training_crops", output_report="dataset_analysis_report.png"):
    classes = ["SCC", "HSIL", "ASC-H", "LSIL", "ASCUS", "NILM", "ENDO", "INFL"]
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory '{dataset_dir}' does not exist. Run wsi_auto_cropper.py first.")
        return
        
    class_counts = {}
    corrupted_count = 0
    empty_count = 0
    dimensions = []
    
    nucleus_sizes = defaultdict(list)
    cell_sizes = defaultdict(list)
    sample_images = defaultdict(list)
    
    print("Analyzing dataset... This may take a few minutes depending on size.\n")
    
    total_images = 0
    
    for c in classes:
        class_dir = os.path.join(dataset_dir, c)
        if not os.path.exists(class_dir):
            class_counts[c] = 0
            continue
            
        files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg'))]
        class_counts[c] = len(files)
        total_images += len(files)
        
        # Select up to 5 random samples for visualization
        if len(files) > 0:
            sample_files = random.sample(files, min(5, len(files)))
            for sf in sample_files:
                img_path = os.path.join(class_dir, sf)
                img = cv2.imread(img_path)
                if img is not None:
                    sample_images[c].append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Analyze a subset (max 100 per class) for speed
        analyze_files = random.sample(files, min(100, len(files))) if files else []
        
        for f in analyze_files:
            img_path = os.path.join(class_dir, f)
            img = cv2.imread(img_path)
            
            # 1. Corrupted Check
            if img is None:
                corrupted_count += 1
                continue
                
            dimensions.append(img.shape[:2])
            
            # 2. Empty/Blank Check
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if np.std(gray) < 5 or np.mean(gray) > 245:
                empty_count += 1
                continue
                
            # 3. Nucleus and Cell Size Estimation via H&E
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hed = rgb2hed(rgb)
                
                hematoxylin = hed[:, :, 0]
                eosin = hed[:, :, 1]
                
                nuc_area = np.sum(hematoxylin > np.percentile(hematoxylin, 80))
                cyto_area = np.sum(eosin > np.percentile(eosin, 80))
                
                nucleus_sizes[c].append(nuc_area)
                cell_sizes[c].append(nuc_area + cyto_area)
            except:
                pass
                
    # --- PRINT TEXT REPORT ---
    print("=========================================")
    print("       DATASET VALIDATION REPORT         ")
    print("=========================================\n")
    
    print(f"Total Images Analyzed: {total_images}")
    print(f"Corrupted Images Detected: {corrupted_count}")
    print(f"Potentially Empty/Blank Images Detected: {empty_count}")
    
    if dimensions:
        avg_h = int(np.mean([d[0] for d in dimensions]))
        avg_w = int(np.mean([d[1] for d in dimensions]))
        print(f"Average Crop Dimensions: {avg_w} x {avg_h} pixels\n")
        
    print("--- Class Distribution & Imbalance ---")
    for c in classes:
        count = class_counts.get(c, 0)
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"[{c:<5}] : {count:<5} images ({percentage:.1f}%)")
        
    if total_images == 0:
        print("\nDataset is empty! No report to generate.")
        return

    # --- GENERATE VISUAL REPORT ---
    fig = plt.figure(figsize=(18, 14))
    fig.canvas.manager.set_window_title('Cervical Dataset Analysis')
    
    # 1. Bar Chart: Class Imbalance
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    counts = [class_counts.get(c, 0) for c in classes]
    bars = ax1.bar(classes, counts, color='teal')
    ax1.set_title("Class Distribution (Imbalance Check)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Number of Images")
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')
        
    # 2. Scatter Plot: Nucleus vs Cell Size
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    colors = plt.cm.get_cmap('tab10', len(classes))
    for i, c in enumerate(classes):
        if c in nucleus_sizes and c in cell_sizes:
            ax2.scatter(cell_sizes[c], nucleus_sizes[c], label=c, alpha=0.6, color=colors(i))
    ax2.set_title("Morphology: Nucleus vs Cell Size", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Estimated Total Cell Area")
    ax2.set_ylabel("Estimated Nucleus Area")
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    
    # 3. Text Statistics Panel
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3.axis('off')
    stats_text = (
        f"DATASET HEALTH SUMMARY\n\n"
        f"Total Valid Images : {total_images}\n"
        f"Corrupted Files    : {corrupted_count}\n"
        f"Empty/Blank Crops  : {empty_count}\n"
        f"Avg Resolution     : {avg_w} x {avg_h}\n\n"
        f"IMBALANCE WARNING:\n"
    )
    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=lambda k: class_counts.get(k) if class_counts.get(k) > 0 else float('inf'))
    if class_counts.get(max_class, 0) > 0 and class_counts.get(min_class, 0) > 0:
        ratio = class_counts[max_class] / class_counts[min_class]
        if ratio > 5:
            stats_text += f"Severe imbalance detected! {max_class} is {ratio:.1f}x larger than {min_class}.\nClass-Weighted Loss MUST be used during training."
        else:
            stats_text += "Dataset is reasonably balanced."
    else:
        stats_text += "Not enough data to calculate imbalance ratio."
        
    ax3.text(0.1, 0.5, stats_text, fontsize=11, va='center', bbox=dict(facecolor='lightgray', alpha=0.5, pad=10))

    # 4. Image Samples Grid
    sample_ax_start = 0
    # Create a grid for the 8 classes (2 rows of 4)
    # We will just plot a few random samples in a new figure or bottom row
    
    plt.tight_layout()
    plt.savefig(output_report, dpi=150)
    print(f"\nVisual report saved successfully to: {output_report}")
    
    # Optional: Display samples in a separate figure
    fig2, axes = plt.subplots(len(classes), 5, figsize=(12, 2 * len(classes)))
    fig2.canvas.manager.set_window_title('Random Class Samples')
    fig2.suptitle("Random Visual Samples (Validation Check)", fontsize=14, fontweight='bold')
    
    for row, c in enumerate(classes):
        imgs = sample_images.get(c, [])
        for col in range(5):
            ax = axes[row, col]
            if col < len(imgs):
                ax.imshow(imgs[col])
            ax.axis('off')
            if col == 0:
                ax.set_title(c, loc='left', fontweight='bold', fontsize=10)
                
    plt.tight_layout()
    plt.savefig("dataset_samples_grid.png", dpi=150)
    print(f"Sample grid saved successfully to: dataset_samples_grid.png")


if __name__ == "__main__":
    # Point this to the output directory of wsi_auto_cropper.py
    analyze_dataset(dataset_dir="training_crops")
