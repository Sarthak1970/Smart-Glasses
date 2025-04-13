import cv2
import torch
import time
import numpy as np
import io
import sqlite3
from transformers import BlipProcessor, BlipForConditionalGeneration
from paddleocr import PaddleOCR
import pyttsx3
from PIL import Image

# Flask MJPEG stream URL
FLASK_IMAGE_URL = "http://192.168.137.126/video_feed"  

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

# Load Models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using Device: {device}")

# TTS Engine
tts_engine = pyttsx3.init()

# BLIP for Image Captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def enhance_frame(frame):
    """Apply sharpening and slight contrast enhancement."""
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    return cv2.filter2D(frame, -1, kernel)

def generate_caption(frame):
    """Generate caption using BLIP for the given frame."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

def perform_ocr(frame):
    """Perform OCR on the frame."""
    ocr_results = ocr.ocr(frame, cls=True)
    extracted_text = []
    for line in ocr_results:
        if line:
            for res in line:
                if res and len(res) > 1:
                    extracted_text.append(res[1][0])
    return "\n".join(extracted_text) if extracted_text else "No Text Detected"

def store_frame(frame, caption, ocr_text):
    """Save the frame and metadata to the database."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    cursor.execute("INSERT INTO Images (timestamp, image_data, caption, ocr_text) VALUES (datetime('now'), ?, ?, ?)",
                   (img_bytes, caption, ocr_text))
    conn.commit()

def speak_text(text):
    """Speak given text using TTS."""
    if text:
        tts_engine.say(text)
        tts_engine.runAndWait()

# Open stream
cap = cv2.VideoCapture(FLASK_IMAGE_URL)
if not cap.isOpened():
    print("Could not connect to video stream.")
    exit()

store_interval = 3  # seconds
last_store_time = time.time()

print("🎥 Streaming video with live captioning & OCR...")

caption = ""
ocr_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        time.sleep(1)
        continue

    frame = enhance_frame(frame)

    # Only update caption & OCR occasionally to avoid lag
    if time.time() - last_store_time >= store_interval:
        caption = generate_caption(frame)
        ocr_text = perform_ocr(frame)
        store_frame(frame, caption, ocr_text)
        print("📝 Auto-saved video frame, caption, and OCR.")
        last_store_time = time.time()

    # Annotate using OpenCV
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Caption: {caption}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if ocr_text != "No Text Detected":
        cv2.putText(display_frame, f"OCR: {ocr_text[:50]}...", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Live Video Feed", cv2.resize(display_frame, (640, 480)))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        store_frame(frame, caption, ocr_text)
        print("Manually saved frame.")
    elif key == ord('t'):
        speak_text(caption)
    elif key == ord('o'):
        speak_text(ocr_text)

# Cleanup
cap.release()
cv2.destroyAllWindows()
conn.close()
tts_engine.stop()
