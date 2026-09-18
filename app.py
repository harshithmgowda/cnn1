import io
import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import onnxruntime as ort

from werkzeug.middleware.proxy_fix import ProxyFix

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cifar10_cnn.onnx")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}
MAX_CONTENT_LENGTH = 4 * 1024 * 1024  # 4 MB max upload

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
    template_folder=os.path.join(BASE_DIR, "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Wrap with ProxyFix for reverse-proxy deployments (Vercel, Nginx)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# CIFAR-10 classes (index 0..9)
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Device info
device_name = "ONNX Runtime (CPU)"

# Load ONNX model
session = None
if os.path.exists(MODEL_PATH):
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
else:
    print(f"WARNING: Model not found at {MODEL_PATH}")


def preprocess_image(image):
    """Resize to 32x32, normalize to [-1, 1], return numpy array [1, 3, 32, 32]."""
    image = image.resize((32, 32))
    img_array = np.array(image, dtype=np.float32) / 255.0
    # Normalize with mean=0.5, std=0.5 per channel
    img_array = (img_array - 0.5) / 0.5
    # HWC -> CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def softmax(x):
    """Compute softmax values."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
@app.route("/index", methods=["GET"])
@app.route("/index.html", methods=["GET"])
@app.route("/api", methods=["GET"])
@app.route("/api/index", methods=["GET"])
@app.route("/api/index.py", methods=["GET"])
def index():
    return render_template("index.html", device_name=device_name)


@app.route("/predict", methods=["GET", "POST"])
@app.route("/api/predict", methods=["GET", "POST"])
def predict():
    # Provide helpful response if opened via GET in browser
    if request.method == "GET":
        return jsonify({
            "status": "online",
            "endpoint": request.path,
            "method": "POST",
            "description": "Send a POST request with an image file under the 'image' field to classify."
        })

    # Validate file present
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request. Please select an image first."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected. Please select an image first."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload an image (png/jpg/jpeg/gif/bmp)."}), 400

    if session is None:
        return jsonify({"error": "Model not loaded. Please ensure cifar10_cnn.onnx exists in models/."}), 500

    filename = secure_filename(file.filename)

    # Read file into PIL safely (no permanent storage)
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "Could not process the image. The file may be corrupted."}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {str(e)}"}), 500

    try:
        # Preprocess
        image_tensor = preprocess_image(image)

        # Inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: image_tensor})
        logits = outputs[0]  # shape: [1, 10]

        probabilities = softmax(logits)

        predicted_idx = int(np.argmax(probabilities, axis=1)[0])
        predicted_label = CLASSES[predicted_idx]
        confidence_pct = float(probabilities[0][predicted_idx]) * 100.0

        # Top-3
        top3_indices = np.argsort(probabilities[0])[::-1][:3]
        top3 = []
        for idx in top3_indices:
            top3.append({
                "label": CLASSES[int(idx)],
                "confidence": round(float(probabilities[0][idx]) * 100.0, 2)
            })

        # Return result
        return jsonify({
            "prediction": predicted_label,
            "confidence": round(confidence_pct, 2),
            "top3": top3,
            "device": device_name
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.errorhandler(404)
def handle_not_found(e):
    # If the user opens an unknown path in browser, serve the UI seamlessly
    if request.accept_mimetypes.accept_html:
        return render_template("index.html", device_name=device_name)
    return jsonify({
        "error": "The requested URL was not found on the server. Please check your spelling and try again."
    }), 404


if __name__ == "__main__":
    app.run(debug=True)
