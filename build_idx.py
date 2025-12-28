import os
import sys

files = sorted(os.listdir(sys.argv[1]))
with open(sys.argv[2], 'w') as f:
    f.writelines(f"{item.split('.')[0]}\n" for item in files)
