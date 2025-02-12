import cv2
import torch
import time
import numpy as np
import io
import sqlite3
import requests
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from transformers import AutoProcessor, AutoModelForCausalLM

# Set the ESP32-CAM Stream URL (Update with correct IP)
ESP32_CAM_URL = "http://192.168.186.243:81/stream"

# Initialize SQLite Database
conn = sqlite3.connect('captured_images.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS Images (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      image_data BLOB
                  )''')
conn.commit()

# Load Florence-2 Model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using Device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# Function to store images in the database
def store_image_in_db(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    cursor.execute("INSERT INTO Images (timestamp, image_data) VALUES (datetime('now'), ?)", (img_bytes,))
    conn.commit()

# OpenCV Video Capture
cap = cv2.VideoCapture(ESP32_CAM_URL)
if not cap.isOpened():
    print("❌ Failed to open ESP32-CAM stream. Check the URL.")
    exit()

store_interval = 15  # Auto-save interval in seconds
last_store_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame from ESP32-CAM.")
        break
    
    try:
        # Convert OpenCV BGR frame to PIL RGB image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Generate caption using Florence-2
        prompt = "<CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        try:
            caption = processor.post_process_generation(
                generated_text,
                task="<CAPTION>",
                image_size=(image.width, image.height)
            )['<CAPTION>']
        except KeyError:
            caption = "No Caption Generated"

        # Draw the caption on the image
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((10, 10), caption, fill="red")

        # Convert back to OpenCV format and display
        frame_with_caption = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame_with_caption = cv2.resize(frame_with_caption, (640, 480))
        cv2.imshow("ESP32-CAM with Caption", frame_with_caption,)

        # Save images automatically at intervals
        if time.time() - last_store_time >= store_interval:
            store_image_in_db(image)
            print("✅ Annotated image stored automatically.")
            last_store_time = time.time()

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            store_image_in_db(image)
            print("✅ Image manually stored.")

    except UnidentifiedImageError:
        print("❌ ESP32-CAM sent an invalid image. Skipping frame.")
        continue

# Cleanup
cap.release()
cv2.destroyAllWindows()
conn.close()
