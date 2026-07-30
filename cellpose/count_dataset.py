import os

# Likely base paths based on previous messages
base_paths = [
    r"C:\Users\Hohin.J\Downloads\202604-20260521T060434Z-3-002\202604",
    r"C:\Users\Hohin.J\Downloads",
    r"C:\Users\Hohin.J\cellpose",
    r"C:\Users\Hohin.J\cellpose\cellpose"
]

classes = ["SCC", "HSIL", "ASC-H", "LSIL", "ASCUS", "NILM", "ENDO", "INFL"]

found_dataset = False
for base in base_paths:
    if not os.path.exists(base): continue
    
    counts = {}
    for c in classes:
        p = os.path.join(base, c)
        if os.path.isdir(p):
            counts[c] = len([f for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
    
    if counts:
        print(f"Found dataset in: {base}")
        for c in classes:
            print(f"{c}: {counts.get(c, 0)} images")
        found_dataset = True
        break

if not found_dataset:
    print("Could not find the dataset folders automatically.")
