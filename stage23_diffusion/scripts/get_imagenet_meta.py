import sys
sys.path.append(".")
import os

imagenet_folder = sys.argv[1]
data_type = sys.argv[2]
savepath = sys.argv[3]

# self.paths = paths_from_folder(self.gt_folder)
is_train = data_type == "train"
if is_train:
    file_list = os.path.join(imagenet_folder, "meta", "train.txt")
else:
    file_list = os.path.join(imagenet_folder, "meta", "test.txt")

os.makedirs(os.path.dirname(savepath), exist_ok=True)
fout = open(savepath, "w")

count = 0
with open(file_list) as fin:
    for line in fin:
        line = line.strip()
        # skip empty line
        if line:
            relative_path = line.split()[0] # "/path/to/image class_id"
            full_path = os.path.join(imagenet_folder, "train" if is_train else "test", relative_path)
            fout.write(f"{full_path}\n")
            count = count + 1

print(f"done, number of files: {count}")
