import cv2
import os
import shutil
from YOLO import YOLO_cropper, YOLO_Interface
from CRNN import CNNR_Interface

BW_img_loc = 'BW_read_img.jpg'
cropped_imgs_loc = 'cropped_words'

def to_black_and_white(image_path, save_path=None, white_thresh=200):
    
    # Read image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if save_path:
        cv2.imwrite(save_path, gray)

    return gray

def clear_folder(folder_path):
    # Create folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created folder '{folder_path}'")
        return

    # Clear existing contents
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove subdirectory
        except Exception as e:
            print(f"⚠️ Failed to delete {file_path}. Reason: {e}")

    print(f"✅ Cleared all contents in '{folder_path}'")

def pipeline(img_path, bw=False):
    to_black_and_white(image_path=img_path, save_path=BW_img_loc)
    detected_text = ""
    path = img_path
    if bw:
        path = BW_img_loc

    clear_folder(cropped_imgs_loc)
    print("done cleaning")

    #YOLO_Interface(path)
    #print("done locating")

    rows = YOLO_cropper(path)
    if not rows:
        return "❌ No text detected."

    # Process rows separately
    line_texts = []
    for row in rows:
        words = []
        for loc in row:
            words.append(CNNR_Interface(loc))
        line_texts.append(" ".join(words))  # join words in a row

    detected_text = "\n".join(line_texts)  # join rows with newlines

    clear_folder(cropped_imgs_loc)
    return detected_text.strip()

if __name__ == '__main__':
    print(pipeline(img_path='BW_read_img.jpg', bw=True))