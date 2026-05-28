import io
from unittest.mock import MagicMock, patch

import pytest

from llm.vhs_schema import VHSInterpretation


@pytest.fixture()
def client():
    with (
        patch("app.download_model", return_value=True),
        patch("app.os.path.exists", return_value=True),
    ):
        from app import app as flask_app

        flask_app.config["TESTING"] = True
        with flask_app.test_client() as c:
            yield c


FAKE_INTERPRETATION = VHSInterpretation(
    vhs_score=9.7,
    normal_range="8.7-10.7",
    interpretation="normal",
    severity="none",
    possible_conditions=[],
    recommendations=["Annual re-check"],
    detailed_explanation="Heart size is within normal limits.",
)


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200


def test_analyze_returns_vhs_score_and_interpretation(client):
    with patch("app.interpret_vhs", return_value=FAKE_INTERPRETATION):
        res = client.post(
            "/api/analyze",
            json={
                "l_value": 55.0,
                "s_value": 42.0,
                "t_value": 60.0,
                "animal_type": "Dog",
            },
        )
    assert res.status_code == 200


def test_extract_no_image_returns_400(client):
    res = client.post("/api/extract")
    assert res.status_code == 400


def test_extract_returns_measurements_and_image(client):
    with (
        patch("app.get_model") as mock_get_model,
        patch("app._predict_from_image", return_value=(55.0, 42.0, 60.0, 9.7, "base64img")),
    ):
        mock_get_model.return_value = (MagicMock(), MagicMock())
        fake_image = (io.BytesIO(b"fake-image-data"), "xray.png")
        res = client.post(
            "/api/extract", data={"image": fake_image}, content_type="multipart/form-data"
        )

    assert res.status_code == 200
    body = res.get_json()
    assert body["l_value"] == 55.0
    assert body["s_value"] == 42.0
    assert body["t_value"] == 60.0
    assert "annotated_image" in body
