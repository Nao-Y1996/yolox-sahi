import os

ROOT = os.path.dirname(os.path.abspath(__file__).replace('src/const', ''))
label_file = os.path.join(ROOT, "src/const/coco_labels.txt")
COCO_LABELS = []
with open(label_file, "r") as f:
    for line in f:
        COCO_LABELS.append(line.strip())

if __name__ == '__main__':
    print(ROOT)
    print(COCO_LABELS)