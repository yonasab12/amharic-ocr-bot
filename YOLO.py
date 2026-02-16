from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

cropped_imgs_loc = 'cropped_words'
confidenceRate = 0.2
iou = 0.2

yolo_model = YOLO('best.pt')

def YOLO_Interface(image_path):
    # 3. Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        exit()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = image_rgb.copy() # Create a copy to draw on

    results = yolo_model.predict(source=image_path, conf=confidenceRate, iou=iou, save=False, show=False)
    
    for r in results:
        if r.boxes: 
            for box in r.boxes:   
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                color = (0, 255, 0)
                if yolo_model.task == 'segment': # Example: different color for segmentation if applicable
                    color = (255, 0, 255) # Magenta

                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)

    # 6. Display the annotated image using Matplotlib
    plt.figure(figsize=(12, 10)) # Adjust figure size for better viewing
    plt.imshow(display_image)
    plt.axis('off') # Hide axes
    plt.title('Model Prediction with Custom Labels')
    plt.show()
    cv2.imwrite('yoloOUTput.jpg',display_image)


def YOLO_cropper(image_path, output_folder="cropped_words", row_tolerance=0.6):
    os.makedirs(output_folder, exist_ok=True)

    results = yolo_model.predict(source=image_path, conf=confidenceRate, iou=iou, save=False, show=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print("⚠️ No boxes detected.")
        return []

    crops_info = []
    for x1, y1, x2, y2 in boxes:
        h = y2 - y1
        yc = (y1 + y2) / 2
        crops_info.append((int(x1), int(y1), int(x2), int(y2), yc, h))

    crops_info.sort(key=lambda b: b[4])  # sort by y

    # Group into rows
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

    # Save crops in row order
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

    print(f"✅ Saved {sum(len(r) for r in saved_paths_rows)} crops grouped into {len(saved_paths_rows)} rows.")
    return saved_paths_rows  # return rows instead of flat list

if __name__ == '__main__':
    path = 'BW_read_img.jpg'
    YOLO_cropper(path)