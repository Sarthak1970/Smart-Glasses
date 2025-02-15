import cv2
import torch
import time
import numpy as np
import io
import sqlite3
import requests
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from transformers import AutoProcessor, AutoModelForCausalLM

ESP32_CAM_URL = "http://192.168.186.243:81/stream"

conn = sqlite3.connect('..\Backend\captured_images.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS Images (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      image_data BLOB,
                      caption TEXT DEFAULT NULL
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

def enhance_image(image):
    """Enhance image by applying sharpening and adjusting brightness/contrast."""
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    kernel = np.array([[0, -0.5, 0],
                   [-0.5, 3, -0.5],
                   [0, -0.5, 0]])

    image_cv = cv2.filter2D(image_cv, -1, kernel)
    
    # Convert to HSV and adjust brightness/contrast
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    image_cv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

def store_image_in_db(image, caption):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    cursor.execute("INSERT INTO Images (timestamp, image_data, caption) VALUES (datetime('now'), ?, ?)", (img_bytes, caption))
    conn.commit()

# Open the MJPEG stream
stream = requests.get(ESP32_CAM_URL, stream=True)
if stream.status_code != 200:
    print("❌ Failed to connect to ESP32-CAM!")
    exit()

bytes_data = b""
store_interval = 3
last_store_time = time.time()

# Process MJPEG stream
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
            image = image.rotate(-90, expand=True)  # Rotate 90 degrees clockwise

            
            image = enhance_image(image)

            prompt = "Describe Image Briefly"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    num_beams=2,
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
            font = ImageFont.truetype("arial.ttf", 14) 
            draw.text((10, 10), caption, fill="red", font=font)

            frame_with_caption = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frame_with_caption = cv2.resize(frame_with_caption, (640, 480))
            cv2.imshow("ESP32-CAM with Caption", frame_with_caption)

            if time.time() - last_store_time >= store_interval:
                store_image_in_db(image, caption)
                print("✅ Annotated image and caption stored automatically.")
                last_store_time = time.time()

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                store_image_in_db(image, caption)
                print("✅ Image and caption manually stored.")

        except UnidentifiedImageError:
            print("❌ ESP32-CAM sent an invalid image. Skipping frame.")
            continue

# Cleanup
cv2.destroyAllWindows()
conn.close()