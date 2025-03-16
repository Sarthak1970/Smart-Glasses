import cv2
import torch
import time
import numpy as np
import io
import sqlite3
import requests
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from transformers import BlipProcessor, BlipForConditionalGeneration
from paddleocr import PaddleOCR
import pyttsx3


ESP32_CAM_URL = "http://192.168.169.243:81/stream"

conn = sqlite3.connect('../Backend/captured_images.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS Images (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      image_data BLOB,
                      caption TEXT DEFAULT NULL,
                      ocr_text TEXT DEFAULT NULL
                  )''')
conn.commit()

# Load BLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

tts_engine = pyttsx3.init()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

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

def store_image_in_db(image, caption, ocr_text):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    cursor.execute("INSERT INTO Images (timestamp, image_data, caption, ocr_text) VALUES (datetime('now'), ?, ?, ?)", 
                   (img_bytes, caption, ocr_text))
    conn.commit()

def speak_text(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

stream = requests.get(ESP32_CAM_URL, stream=True)
if stream.status_code != 200:
    print("❌ Failed to connect to ESP32-CAM!")
    exit()

bytes_data = b""
store_interval = 3
last_store_time = time.time()

for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')  

    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]  
        bytes_data = bytes_data[b+2:]  

        try:
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = enhance_image(image)

            # Generate caption using BLIP
            caption = generate_caption(image)

            # OCR Processing
            ocr_results = ocr.ocr(np.array(image), cls=True)
            if ocr_results and isinstance(ocr_results, list):
                ocr_text = "\n".join([res[1][0] for line in ocr_results if line for res in line if res and len(res) > 1])
            else:
                ocr_text = "No Text Detected"

            # Annotate image with caption and OCR text
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("arial.ttf", 14)
            draw.text((10, 10), caption, fill="red", font=font)
            if ocr_text != "No Text Detected":
                draw.text((10, 30), "OCR: " + ocr_text[:50] + "...", fill="blue", font=font)


            frame_with_caption = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frame_with_caption = cv2.resize(frame_with_caption, (640, 480))
            cv2.imshow("ESP32-CAM with Caption & OCR", frame_with_caption)

            # Store image at regular intervals
            if time.time() - last_store_time >= store_interval:
                store_image_in_db(image, caption, ocr_text)
                print("✅ Annotated image, caption, and OCR text stored automatically.")
                last_store_time = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                store_image_in_db(image, caption, ocr_text)
                print("✅ Image, caption, and OCR text manually stored.")
            elif key == ord('t'):
                speak_text(caption)

        except UnidentifiedImageError:
            print("❌ ESP32-CAM sent an invalid image. Skipping frame.")
            continue

cv2.destroyAllWindows()
conn.close()
