import gradio as gr
import tempfile
import os
import shutil
from OCR import pipeline  # your existing pipeline

def ocr_function(image):
    # Save uploaded image to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    try:
        # Run OCR (adjust bw=True if needed)
        text = pipeline(temp_path, bw=True)
    except Exception as e:
        text = f"Error: {str(e)}"
    finally:
        os.unlink(temp_path)

    return text

# Create Gradio interface
iface = gr.Interface(
    fn=ocr_function,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Recognized Text"),
    title="Amharic OCR",
    description="Upload an image containing Amharic text, and the model will extract it."
)

if __name__ == "__main__":
    iface.launch()
