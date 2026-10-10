import os
import traceback

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Anchor every path to this file's own folder, not wherever the terminal
# happens to be pointed - avoids "file not found" errors from launching
# the script in different ways (VS Code Run button, different cwd, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = 224

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

for folder in (UPLOAD_FOLDER, GRADCAM_FOLDER):
    os.makedirs(folder, exist_ok=True)


# ============================================================
# Grad-CAM helpers (kept in this same file - no separate module
# to import, so there's no risk of running a stale cached copy)
# ============================================================

def ensure_model_built(model, input_shape=(224, 224, 3)):
    """
    Some Sequential models saved without an explicit Input(shape=...) layer
    don't know their own output tensor until called at least once.
    Running one dummy prediction fixes this before Grad-CAM needs model.outputs.
    """
    dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
    model(dummy_input)
    return model


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def build_grad_model(model, last_conv_layer_name, input_shape=(224, 224, 3)):
    """
    Rebuilds a Grad-CAM sub-model by explicitly replaying every layer on a
    fresh, real Input() tensor. This avoids a known issue where reusing
    model.get_layer(...).output / model.outputs[0] from a Sequential model
    that was only ever called eagerly (not built via a symbolic Input)
    produces a graph that looks fine for prediction, but gives None
    gradients when GradientTape tries to trace back through it.
    """
    img_input = tf.keras.Input(shape=input_shape)
    x = img_input
    conv_layer_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_layer_output = x
    if conv_layer_output is None:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found while rebuilding the graph.")
    return tf.keras.models.Model(inputs=img_input, outputs=[conv_layer_output, x])


def make_gradcam_heatmap(img_array, grad_model, pred_index=None):

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        probabilities = predictions.numpy()[0]
        if pred_index is None:
            pred_index = int(np.argmax(probabilities))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index, probabilities


def overlay_heatmap(original_bgr_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_bgr_img.shape[1], original_bgr_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(colored_heatmap, alpha, original_bgr_img, 1 - alpha, 0)


# ============================================================
# Model loading (runs once at startup)
# ============================================================

FIVE_LEVEL_LABELS = [
    "No DR (Level 0)",
    "Mild NPDR (Level 1)",
    "Moderate NPDR (Level 2)",
    "Severe NPDR (Level 3)",
    "Proliferative DR (Level 4)",
]
BINARY_LABELS = ["Diabetic Retinopathy Detected", "No Diabetic Retinopathy Detected"]

MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model = ensure_model_built(model, input_shape=(IMG_SIZE, IMG_SIZE, 3))
LAST_CONV_LAYER = find_last_conv_layer(model)
GRAD_MODEL = build_grad_model(model, LAST_CONV_LAYER, input_shape=(IMG_SIZE, IMG_SIZE, 3))
NUM_CLASSES = model.output_shape[-1]

if NUM_CLASSES == 5:
    CLASS_LABELS = FIVE_LEVEL_LABELS
    def is_severity_referable(level):
        return level >= 2
elif NUM_CLASSES == 2:
    CLASS_LABELS = BINARY_LABELS
    def is_severity_referable(level):
        return level == 0
else:
    CLASS_LABELS = [f"Class {i}" for i in range(NUM_CLASSES)]
    def is_severity_referable(level):
        return False

print(f"Loaded model. Grad-CAM layer: '{LAST_CONV_LAYER}'. "
      f"Output shape: {model.output_shape} -> {NUM_CLASSES}-class labels: {CLASS_LABELS}")


# ============================================================
# Flask routes
# ============================================================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Could not read image at {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_bgr, np.expand_dims(img_normalized, axis=0)


def predict_and_explain(path, filename):
    original_bgr, img_batch = preprocess_image(path)
    heatmap, pred_index, probabilities = make_gradcam_heatmap(img_batch, GRAD_MODEL)

    overlaid = overlay_heatmap(original_bgr, heatmap)
    gradcam_filename = f"gradcam_{filename}"
    gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_path, overlaid)

    confidence = float(probabilities[pred_index])
    severity_level = int(pred_index)
    is_referable = is_severity_referable(severity_level)

    probability_breakdown = [
        {"label": CLASS_LABELS[i] if i < len(CLASS_LABELS) else f"Class {i}",
         "probability": float(p)}
        for i, p in enumerate(probabilities)
    ]

    return {
        "diagnosis": CLASS_LABELS[severity_level] if severity_level < len(CLASS_LABELS) else f"Class {severity_level}",
        "severity_level": severity_level,
        "confidence": round(confidence * 100, 1),
        "is_referable": is_referable,
        "probability_breakdown": probability_breakdown,
        "gradcam_image": url_for("static", filename=f"gradcam/{gradcam_filename}"),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def serve_uploaded_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/predict", methods=["POST"])
def upload_file():
    print(f"DEBUG: request.files keys = {list(request.files.keys())}")
    print(f"DEBUG: request.form keys = {list(request.form.keys())}")

    if "file" not in request.files:
        print("DEBUG: hit 'No file part' branch - request.files has no 'file' key")
        return render_template("index.html", message="No file part")

    file = request.files["file"]
    print(f"DEBUG: file.filename = '{file.filename}'")
    if file.filename == "":
        print("DEBUG: hit 'No selected file' branch - filename is empty string")
        return render_template("index.html", message="No selected file")

    if not (file and allowed_file(file.filename)):
        print(f"DEBUG: hit 'Unsupported file type' branch for filename '{file.filename}'")
        return render_template("index.html", message="Unsupported file type")

    print("DEBUG: passed all checks, proceeding to prediction...")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        result = predict_and_explain(file_path, filename)
    except Exception as exc:
        print("=" * 60)
        print("PREDICTION FAILED - full traceback below:")
        traceback.print_exc()
        print("=" * 60)
        return render_template("index.html", message=f"Error processing image: {exc}")

    return render_template(
        "predict.html",
        diagnosis=result["diagnosis"],
        severity_level=result["severity_level"],
        confidence=result["confidence"],
        is_referable=result["is_referable"],
        probability_breakdown=result["probability_breakdown"],
        user_image=url_for("serve_uploaded_image", filename=filename),
        gradcam_image=result["gradcam_image"],
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)