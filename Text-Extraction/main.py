import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt
import cv2
import sqlite3
import time
import io

# SQLite setup
conn = sqlite3.connect('captured_images.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS Images (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      image_data BLOB
                  )''')
conn.commit()

def store_image_in_db(image):
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    # Insert image into database
    cursor.execute("INSERT INTO Images (timestamp, image_data) VALUES (datetime('now'), ?)", (img_bytes,))
    conn.commit()

# Device setup
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

plt.ion()
fig, ax = plt.subplots(figsize=(10, 10))

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Store image every 15 seconds
    if time.time() - start_time >= 15:
        store_image_in_db(image)
        print("Image stored in the database.")
        start_time = time.time()

    # Prepare inputs for the model
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

    # Handle post-processing safely
    try:
        parsed_answer = processor.post_process_generation(
            generated_text,
            task="<CAPTION>",
            image_size=(image.width, image.height)
        )['<CAPTION>']
    except KeyError:
        parsed_answer = "No Caption Generated"

    # Display the image with the caption
    ax.clear()
    ax.imshow(image)
    ax.axis('off')
    ax.text(10, 10, parsed_answer, fontsize=12, color='red', alpha=0.7)

    plt.draw()
    plt.pause(0.01)

cap.release()
plt.ioff()
plt.show()

conn.close()
