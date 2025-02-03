import sqlite3
from PIL import Image
import io

def retrieve_images_by_partial_timestamp(partial_timestamp):
    conn = sqlite3.connect('captured_images.db')
    cursor = conn.cursor()

    cursor.execute("SELECT timestamp, image_data FROM Images WHERE timestamp LIKE ?", (f"%{partial_timestamp}%",))
    rows = cursor.fetchall()

    if not rows:
        print("No images found for the given timestamp.")
    else:
        for idx, (timestamp, img_data) in enumerate(rows):
            img = Image.open(io.BytesIO(img_data))
            img.show(title=f"Image {idx + 1} - {timestamp}")

    conn.close()

if __name__ == "__main__":
    print("Enter partial timestamp in one of the following formats:")
    print("- 'YYYY-MM-DD'for a specific date")
    print("- 'HH:MM' for a specific time")
    print("- 'YYYY-MM-DD HH:MM' for precise matching")
    timestamp_input = input("Enter partial timestamp: ")
    retrieve_images_by_partial_timestamp(timestamp_input)
