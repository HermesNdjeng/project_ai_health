import base64
import io
import os

import matplotlib
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from pydantic import BaseModel, ValidationError

from inference.inference import get_transform, load_model, visualize_prediction_measurements
from llm.vhs_chain import interpret_vhs
from utils.download_model import download_model
from utils.logging_utils import logger

load_dotenv()
matplotlib.use("Agg")
app = Flask(__name__)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/<path:path>", methods=["OPTIONS"])
def options_preflight(path):
    return "", 204


# ── Model initialisation ──────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "best_model_efb7.pt")

if not os.path.exists(MODEL_PATH):
    logger.info("Model not found, downloading…")
    downloaded = download_model()
    if not downloaded:
        logger.error("Model download failed.")

_model = None
_transform = None


def get_model():
    global _model, _transform
    if _model is None:
        _model = load_model(MODEL_PATH)
        _transform = get_transform(resized_image_size=300)
    return _model, _transform


# ── Schemas ───────────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    animal_type: str = "Dog"
    breed: str | None = None
    age: float | None = None
    weight: float | None = None
    sex: str | None = None


class ManualAnalysisRequest(BaseModel):
    l_value: float
    s_value: float
    t_value: float


# ── Utilities ─────────────────────────────────────────────────────────────────
def _predict_from_image(
    image_path: str, model, transform, device
) -> tuple[float, float, float, float, str]:
    fig, ax = plt.subplots(figsize=(10, 10))
    l_value, s_value, t_value, vhs = visualize_prediction_measurements(
        image_path=image_path,
        model=model,
        transform=transform,
        device=device,
        ax=ax,
        resized_image_size=300,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return float(l_value), float(s_value), float(t_value), float(vhs), image_b64


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/extract")
def extract():
    """Extract L, S, T measurements from a radiograph image. No LLM call."""
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    temp_path = f"/tmp/vhs_upload_{file.filename}"
    file.save(temp_path)
    try:
        model, transform = get_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        l_value, s_value, t_value, vhs, image_b64 = _predict_from_image(
            temp_path, model, transform, device
        )
        return jsonify({
            "l_value": round(l_value, 2),
            "s_value": round(s_value, 2),
            "t_value": round(t_value, 2),
            "vhs_score": round(vhs, 2),
            "annotated_image": image_b64,
        })
    except Exception as exc:
        logger.error(f"Extraction error: {exc}")
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/analyze")
def analyze():
    """Run VHS interpretation from L, S, T values and patient data."""
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    try:
        payload = ManualAnalysisRequest.model_validate(data)
        patient = PatientData.model_validate(data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 422

    try:
        interp = interpret_vhs(
            l_value=payload.l_value, s_value=payload.s_value, t_value=payload.t_value,
            **patient.model_dump(),
        )
    except Exception as exc:
        logger.error(f"LLM error: {exc}")
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "vhs_score": round(interp.vhs_score, 2),
        "interpretation": interp.model_dump(),
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)
