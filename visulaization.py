import cv2
import pickle
import os
import torch

# Load predictions
with open("training_predictions2.pkl", "rb") as f:
    prediction_log = pickle.load(f)

# Set file_name from the keys in prediction_log
file_name = "000000000308.txt_5"  # change this to a key from your log

# Strip "_3" (object index) to get base file name
base_name, obj_index = file_name.rsplit("_", 1)
obj_index = int(obj_index)

image_path = os.path.join("archive/coco128/images/train2017", base_name.replace(".txt", ".jpg"))
label_path = os.path.join("archive/coco128/labels/train2017", base_name)

# Load image
img = cv2.imread(image_path)
if img is None:
    print(f"Image not found: {image_path}")
    exit()

h_img, w_img = img.shape[:2]

# Load the corresponding GT line (based on index from suffix)
with open(label_path) as f:
    lines = f.readlines()
    if obj_index >= len(lines):
        print(f"Object index {obj_index} out of range in label file.")
        exit()
    parts = lines[obj_index].strip().split()
    cx, cy, w, h = map(float, parts[1:])

x1_gt = int((cx - w / 2) * w_img)
y1_gt = int((cy - h / 2) * h_img)
x2_gt = int((cx + w / 2) * w_img)
y2_gt = int((cy + h / 2) * h_img)

# All epochs available
epochs = sorted(prediction_log[file_name].keys())
print(epochs)
if len(epochs) <= 1:
    print(f"Not enough epochs for trackbar. Found {len(epochs)}.")
    exit()

# Trackbar update function
def update(val):
    epoch = epochs[val]
    pred = prediction_log[file_name][epoch]
    scale = torch.tensor([w_img, h_img, w_img, h_img], dtype=pred.dtype)
    x1, y1, x2, y2 = (pred * scale).int().tolist()

    img_copy = img.copy()
    cv2.rectangle(img_copy, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 255, 0), 2)  # GT box (green)
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)              # Pred box (red)
    cv2.putText(img_copy, f"Epoch: {epoch}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Optional: legend
    cv2.rectangle(img_copy, (10, 60), (20, 70), (0, 255, 0), -1)
    cv2.putText(img_copy, "GT", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.rectangle(img_copy, (10, 90), (20, 100), (0, 0, 255), -1)
    cv2.putText(img_copy, "Pred", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Prediction Over Epochs", img_copy)

# Display setup
cv2.namedWindow("Prediction Over Epochs")
cv2.createTrackbar("Epoch", "Prediction Over Epochs", 0, len(epochs) - 1, update)
update(0)

# Event loop for clean exit
while True:
    key = cv2.waitKey(1) & 0xFF
    if cv2.getWindowProperty("Prediction Over Epochs", cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
