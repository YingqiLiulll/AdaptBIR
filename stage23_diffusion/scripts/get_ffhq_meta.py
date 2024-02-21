import sys
sys.path.append(".")
import os

ffhq_folder = sys.argv[1]
savepath = sys.argv[2]

os.makedirs(os.path.dirname(savepath), exist_ok=True)
fout = open(savepath, "w")

count = 0
files = os.listdir(ffhq_folder)
for file in files:
    filepath = os.path.abspath(os.path.join(ffhq_folder, file))
    fout.write(f"{filepath}\n")
    count += 1

print(f"done, number of files: {count}")
