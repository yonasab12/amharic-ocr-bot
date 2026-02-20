# OCR.py (modified)
import cv2
import tempfile
import shutil
import os
from YOLO import YOLO_cropper
from CRNN import CNNR_Interface

def to_black_and_white(image_path, save_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_path, gray)
    return gray

def pipeline(img_path, bw=False):
    # Create a temporary directory for all intermediate files
    temp_dir = tempfile.mkdtemp()
    bw_path = os.path.join(temp_dir, 'bw.jpg')
    cropped_dir = os.path.join(temp_dir, 'cropped_words')

    # Convert to BW if requested
    if bw:
        to_black_and_white(img_path, bw_path)
        process_path = bw_path
    else:
        process_path = img_path

    # Run YOLO cropping
    rows = YOLO_cropper(process_path, cropped_dir)
    if not rows:
        shutil.rmtree(temp_dir)
        return "❌ No text detected."

    # Process each crop with CRNN
    line_texts = []
    for row in rows:
        words = []
        for loc in row:
            words.append(CNNR_Interface(loc))
        line_texts.append(" ".join(words))

    detected_text = "\n".join(line_texts)

    # Clean up temp dir
    shutil.rmtree(temp_dir)
    return detected_text.strip()
