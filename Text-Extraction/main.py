import cv2
import torch
import time
import numpy as np
import io
import sqlite3
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from transformers import BlipProcessor, BlipForConditionalGeneration
from paddleocr import PaddleOCR
import pyttsx3

# Database Setup
conn = sqlite3.connect('../Backend/captured_images.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS Images (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      image_data BLOB,
                      caption TEXT DEFAULT NULL,
                      ocr_text TEXT DEFAULT NULL
                  )''')
conn.commit()

# Load BLIP Model for Image Captioning
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using Device: {device}")

# Initialize TTS Engine
tts_engine = pyttsx3.init()

# Load Image Captioning Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Initialize OCR Engine
ocr = PaddleOCR(use_angle_cls=True, lang='en')


def enhance_image(image):
    """Enhance image by applying sharpening and adjusting brightness/contrast."""
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    image_cv = cv2.filter2D(image_cv, -1, kernel)

    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))


def generate_caption(image):
    """Generate a caption for the given image using BLIP."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


def perform_ocr(image):
    """Extract text from the image using OCR (PaddleOCR)."""
    ocr_results = ocr.ocr(np.array(image), cls=True)
    if ocr_results and isinstance(ocr_results, list):
        extracted_text = []
        for line in ocr_results:
            if line:
                for res in line:
                    if res and len(res) > 1:
                        extracted_text.append(res[1][0])
        return "\n".join(extracted_text) if extracted_text else "No Text Detected"
    return "No Text Detected"


def store_image_in_db(image, caption, ocr_text):
    """Store image, caption, and OCR text in the SQLite database."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    cursor.execute("INSERT INTO Images (timestamp, image_data, caption, ocr_text) VALUES (datetime('now'), ?, ?, ?)",
                   (img_bytes, caption, ocr_text))
    conn.commit()


def speak_text(text):
    """Convert text to speech using pyttsx3."""
    if text:
        tts_engine.say(text)
        tts_engine.runAndWait()


# Start Webcam Capture
cap = cv2.VideoCapture(0)  # Capture from the default webcam
if not cap.isOpened():
    print("❌ Failed to access the webcam!")
    exit()

store_interval = 3  # Interval for auto-storing images
last_store_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture image from webcam.")
        break

    try:
        # Convert frame to PIL format and enhance it
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = enhance_image(image)

        # Generate Caption & OCR text
        caption = generate_caption(image)
        ocr_text = perform_ocr(image)

        # Annotate Image
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 14)
        draw.text((10, 10), f"Caption: {caption}", fill="red", font=font)
        if ocr_text != "No Text Detected":
            draw.text((10, 30), f"OCR: {ocr_text[:50]}...", fill="blue", font=font)

        # Convert Image back for Display
        frame_with_caption = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame_with_caption = cv2.resize(frame_with_caption, (640, 480))
        cv2.imshow("Webcam with Caption & OCR", frame_with_caption)

        # Auto-save images at intervals
        if time.time() - last_store_time >= store_interval:
            store_image_in_db(image, caption, ocr_text)
            print("✅ Auto-saved Image, Caption, and OCR Text.")
            last_store_time = time.time()

        # User Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('s'):  # Press 's' to save manually
            store_image_in_db(image, caption, ocr_text)
            print("✅ Manually saved Image, Caption, and OCR Text.")
        elif key == ord('t'):  # Press 't' to speak caption
            speak_text(caption)
        elif key == ord('o'):  # Press 'o' to speak OCR text
            speak_text(ocr_text)

    except UnidentifiedImageError:
        print("❌ Invalid image received. Skipping frame.")
        continue

cap.release()
cv2.destroyAllWindows()
conn.close()
tts_engine.stop()