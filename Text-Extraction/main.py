import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import sqlite3
import io
import time
import numpy as np

conn = sqlite3.connect('captured_images.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS Images (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      image_data BLOB
                  )''')
conn.commit()

def store_image_in_db(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    cursor.execute("INSERT INTO Images (timestamp, image_data) VALUES (datetime('now'), ?)", (img_bytes,))
    conn.commit()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using Device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device.")

store_interval = 15  
last_store_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((10, 10), caption, fill="red")

    frame_with_caption = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    cv2.imshow("Smart Glasses", frame_with_caption)

    if time.time() - last_store_time >= store_interval:
        store_image_in_db(image)
        print("Annotated image automatically stored.")
        last_store_time = time.time()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        store_image_in_db(image)
        print("Annotated image manually stored in the database.")

cap.release()
cv2.destroyAllWindows()
conn.close()
