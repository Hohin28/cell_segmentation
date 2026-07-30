import tifffile
import sys

path = r'C:\Users\Hohin.J\Downloads\202604-20260521T060434Z-3-002\202604\ASCUS\428.26 Ecto.tif'
try:
    with tifffile.TiffFile(path) as tif:
        print("Pages:", len(tif.pages))
        print("Shape:", tif.pages[0].shape)
        print("Dtype:", tif.pages[0].dtype)
except Exception as e:
    print("Error:", e)
