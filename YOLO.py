# YOLO.py (modified)
import os
import cv2
import numpy as np
from ultralytics import YOLO

yolo_model = YOLO('best.pt')
confidenceRate = 0.2
iou = 0.2

def YOLO_cropper(image_path, output_folder, row_tolerance=0.6):
    os.makedirs(output_folder, exist_ok=True)

    results = yolo_model.predict(source=image_path, conf=confidenceRate, iou=iou, save=False, show=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return []

    crops_info = []
    for x1, y1, x2, y2 in boxes:
        h = y2 - y1
        yc = (y1 + y2) / 2
        crops_info.append((int(x1), int(y1), int(x2), int(y2), yc, h))

    crops_info.sort(key=lambda b: b[4])  # sort by y

    # Group into rows (same logic as before)
    rows = []
    current_row = [crops_info[0]]
    for box in crops_info[1:]:
        _, _, _, _, yc, h = box
        row_yc = np.median([b[4] for b in current_row])
        row_h = np.median([b[5] for b in current_row])
        if abs(yc - row_yc) <= row_tolerance * row_h:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    rows.append(current_row)

    # Sort each row left-to-right
    for row in rows:
        row.sort(key=lambda b: b[0])

    image = cv2.imread(image_path)
    saved_paths_rows = []
    for row_idx, row in enumerate(rows, start=1):
        saved_row = []
        for word_idx, (x1, y1, x2, y2, _, _) in enumerate(row, start=1):
            crop = image[y1:y2, x1:x2]
            save_path = os.path.join(output_folder, f"row{row_idx}_word{word_idx}.jpg")
            cv2.imwrite(save_path, crop)
            saved_row.append(save_path)
        saved_paths_rows.append(saved_row)

    return saved_paths_rows
