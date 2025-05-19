import os
import cv2
import random
import numpy as np

# Paths
label_dir = "archive/coco128/labels/train2017"
image_dir = "archive/coco128/images/train2017"
output_image_dir = "plots"
output_label_dir = "archive/coco128/purturbed_labels/train2017"

# Settings
thickness = 2
noise_min = 0.02
noise_max = 0.04
color_gt = (0, 255, 0)     # Green for ground truth
color_pred = (255, 0, 0)   # Red for perturbed box

# Create output folders
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Random +/- noise function
def random_operation(val, noise):
    return val + noise if random.choice([True, False]) else val - noise

# Main loop
for label_file in os.listdir(label_dir):
    label_path = os.path.join(label_dir, label_file)
    image_path = os.path.join(image_dir, label_file.replace(".txt", ".jpg"))

    if not os.path.exists(image_path):
        print(f"Skipping missing image: {image_path}")
        continue

    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]
    print(h_img,w_img)
    perturbed_lines = []

    with open(label_path, "r",encoding='utf8',errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        try:
            class_id, cx, cy, w, h = line.strip().split()
            cx, cy, w, h = map(float, (cx, cy, w, h))

            # Draw GT box
            gt_xmin = int((cx - w/2) * w_img)
            gt_ymin = int((cy - h/2) * h_img)
            gt_xmax = int((cx + w/2) * w_img)
            gt_ymax = int((cy + h/2) * h_img)
            cv2.rectangle(img, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), color_gt, thickness)

            # Add random noise
            res = [random.uniform(noise_min, noise_max) for _ in range(4)]
            new_cx = np.clip(random_operation(cx, res[0]), 0, 1)
            new_cy = np.clip(random_operation(cy, res[1]), 0, 1)
            new_w  = np.clip(random_operation(w, res[2]), 0.01, 1)
            new_h  = np.clip(random_operation(h, res[3]), 0.01, 1)

            # Draw predicted box
            xmin = int((new_cx - new_w/2) * w_img)
            ymin = int((new_cy - new_h/2) * h_img)
            xmax = int((new_cx + new_w/2) * w_img)
            ymax = int((new_cy + new_h/2) * h_img)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_pred, thickness)

            # Save new label
            perturbed_lines.append(f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n")

        except Exception as e:
            print(f"Error in {label_file}: {e}")
            continue

    # Save annotated image
    output_img_path = os.path.join(output_image_dir, os.path.basename(image_path))
    cv2.imwrite(output_img_path, img)

    # Save new label
    output_label_path = os.path.join(output_label_dir, label_file)
    with open(output_label_path, "w") as f:
        f.writelines(perturbed_lines)

print("✅ All images processed and saved with perturbed labels.")