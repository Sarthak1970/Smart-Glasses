import time
import requests
import threading
from luma.core.interface.serial import spi, noop
from luma.oled.device import ssd1309
from luma.core.render import canvas
from PIL import ImageFont

from flask import Flask, Response
from picamera2 import Picamera2
import cv2

# ---------- OLED INIT ----------
serial = spi(port=0, device=0, gpio=noop(), bus_speed_hz=8000000)
device = ssd1309(serial, rotate=0)

# Load a font
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except:
    font = None

# ---------- FLASK APP ----------
app = Flask(__name__)

camera = Picamera2()camera.configure(camera.create_video_configuration(main={"size": (640, 480)}))
camera.start()

# ---------- FUNCTIONS ----------

def get_caption():
    """Fetch the latest image caption from API."""
    try:
        res = requests.get("http://192.168.143.211:5000/api/images/latest", timeout=3)
        data = res.json()
        return data.get('caption', 'No Caption')
    except Exception as e:
        print("❌ API Error:", e)
        return "API Error!"

def display_text(msg):
    """Display wrapped text on OLED screen (max 2 lines)."""
    with canvas(device) as draw:
        msg = msg[:40]  # Limit to 2 lines
        line1 = msg[:20]
        line2 = msg[20:]
        draw.text((0, 0), line1, fill="white", font=font)
        draw.text((0, 16), line2, fill="white", font=font)

def caption_loop():
    """Thread loop to keep updating OLED every few seconds."""
    while True:
        caption = get_caption()
        print("🖥 Showing caption on OLED:", caption)
        display_text(caption)
        time.sleep(5)  # Update interval


def gen_frames():
    """Generator to yield video frames as MJPEG stream."""
    while True:
        frame = camera.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return "<h1>Pi Camera Stream</h1><img src='/video_feed'>"


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------- MAIN ----------
if __name__ == '__main__':
    # Start OLED display update thread
    threading.Thread(target=caption_loop, daemon=True).start()

    # Start Flask server
    app.run(host='0.0.0.0', port=80, debug=False)