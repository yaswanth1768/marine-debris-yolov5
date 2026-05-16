import os

import torch
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

UPLOAD_DIR = "static/upload"
OUTPUT_DIR = "static/output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 Starting Flask app...")

# Load model AFTER Flask starts (safe)
model = None


def load_model():
    global model
    if model is None:
        print("🧠 Loading YOLOv5 model...")
        model = torch.hub.load(".", "custom", path="runs/train/exp8/weights/best.pt", source="local")
        print("✅ Model loaded")


@app.route("/", methods=["GET", "POST"])
def index():
    global model
    detected = False

    if model is None:
        load_model()

    if request.method == "POST":
        print("📩 POST request received")

        # Clear old files
        for d in [UPLOAD_DIR, OUTPUT_DIR]:
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))

        file = request.files["image"]
        upload_path = os.path.join(UPLOAD_DIR, "input.jpg")
        file.save(upload_path)

        img = Image.open(upload_path)
        results = model(img)
        results.save(save_dir=OUTPUT_DIR)

        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".jpg"):
                os.rename(os.path.join(OUTPUT_DIR, f), os.path.join(OUTPUT_DIR, "result.jpg"))
                break

        detected = True

    return render_template("index.html", detected=detected)


if __name__ == "__main__":
    print("🌐 Running Flask on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
