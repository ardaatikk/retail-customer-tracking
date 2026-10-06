from unittest.mock import MagicMock

import numpy as np
import pytest

import src.detection.detector as detector_module
from src.detection.detector import (
    Detection,
    PersonDetector,
)


def create_mock_box(
    coordinates: list[float],
    confidence: float,
    class_id: int,
) -> MagicMock:
    box = MagicMock()

    box.xyxy[0].detach.return_value.cpu.return_value.tolist.return_value = (
        coordinates
    )

    box.conf[0].detach.return_value.cpu.return_value.item.return_value = (
        confidence
    )

    box.cls[0].detach.return_value.cpu.return_value.item.return_value = (
        class_id
    )

    return box


def test_detector_initialization(
    monkeypatch,
) -> None:
    mock_model = MagicMock()

    mock_yolo = MagicMock(
        return_value=mock_model
    )

    monkeypatch.setattr(
        detector_module,
        "YOLO",
        mock_yolo,
    )

    detector = PersonDetector(
        model_name="test-model.pt",
        confidence_threshold=0.40,
    )

    mock_yolo.assert_called_once_with(
        "test-model.pt"
    )

    assert detector.model is mock_model
    assert (
        detector.confidence_threshold
        == pytest.approx(0.40)
    )


def test_detect_returns_detection(
    monkeypatch,
) -> None:
    mock_model = MagicMock()

    monkeypatch.setattr(
        detector_module,
        "YOLO",
        MagicMock(return_value=mock_model),
    )

    mock_box = create_mock_box(
        coordinates=[
            100.0,
            200.0,
            300.0,
            600.0,
        ],
        confidence=0.90,
        class_id=0,
    )

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_model.predict.return_value = [
        mock_result
    ]

    detector = PersonDetector(
        confidence_threshold=0.40
    )

    frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    detections = detector.detect(
        frame
    )

    assert len(detections) == 1

    detection = detections[0]

    assert isinstance(
        detection,
        Detection,
    )

    assert detection.x1 == pytest.approx(
        100.0
    )
    assert detection.y1 == pytest.approx(
        200.0
    )
    assert detection.x2 == pytest.approx(
        300.0
    )
    assert detection.y2 == pytest.approx(
        600.0
    )
    assert (
        detection.confidence
        == pytest.approx(0.90)
    )
    assert detection.class_id == 0


def test_detect_multiple_people(
    monkeypatch,
) -> None:
    mock_model = MagicMock()

    monkeypatch.setattr(
        detector_module,
        "YOLO",
        MagicMock(return_value=mock_model),
    )

    first_box = create_mock_box(
        coordinates=[
            100.0,
            200.0,
            300.0,
            600.0,
        ],
        confidence=0.90,
        class_id=0,
    )

    second_box = create_mock_box(
        coordinates=[
            400.0,
            100.0,
            500.0,
            500.0,
        ],
        confidence=0.80,
        class_id=0,
    )

    mock_result = MagicMock()
    mock_result.boxes = [
        first_box,
        second_box,
    ]

    mock_model.predict.return_value = [
        mock_result
    ]

    detector = PersonDetector()

    frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    detections = detector.detect(
        frame
    )

    assert len(detections) == 2

    assert detections[0].confidence == pytest.approx(
        0.90
    )

    assert detections[1].confidence == pytest.approx(
        0.80
    )


def test_detect_returns_empty_list_when_no_results(
    monkeypatch,
) -> None:
    mock_model = MagicMock()

    monkeypatch.setattr(
        detector_module,
        "YOLO",
        MagicMock(return_value=mock_model),
    )

    mock_model.predict.return_value = []

    detector = PersonDetector()

    frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    detections = detector.detect(
        frame
    )

    assert detections == []


def test_detect_returns_empty_list_when_boxes_are_none(
    monkeypatch,
) -> None:
    mock_model = MagicMock()

    monkeypatch.setattr(
        detector_module,
        "YOLO",
        MagicMock(return_value=mock_model),
    )

    mock_result = MagicMock()
    mock_result.boxes = None

    mock_model.predict.return_value = [
        mock_result
    ]

    detector = PersonDetector()

    frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    detections = detector.detect(
        frame
    )

    assert detections == []


def test_predict_uses_person_class_and_threshold(
    monkeypatch,
) -> None:
    mock_model = MagicMock()

    monkeypatch.setattr(
        detector_module,
        "YOLO",
        MagicMock(return_value=mock_model),
    )

    mock_model.predict.return_value = []

    detector = PersonDetector(
        confidence_threshold=0.42
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    detector.detect(
        frame
    )

    mock_model.predict.assert_called_once_with(
        source=frame,
        classes=[0],
        conf=0.42,
        verbose=False,
    )