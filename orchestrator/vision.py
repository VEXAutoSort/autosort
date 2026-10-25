"""Vision: count parts on the table (overhead camera) and classify the part in the cup.

count_parts() is a contour count on a locked-exposure frame — the same recipe as
autosort/perception.py, kept deliberately simple: blur, Otsu threshold, open,
count contours above a minimum area. Nothing moves while a frame is taken, so
motion blur is not a concern; locked exposure is (vision-tools/2_lock_camera_linux.sh).

Two classifier backends behind one `classify(frame) -> (label, confidence)`:
    OnnxClassifier     YOLOv8n-cls exported to ONNX (onnxruntime)
    CatalogClassifier  the repo's existing measure-and-match (autosort/classification)
cv2 / numpy / onnxruntime are imported lazily so --sim and the tests need none of them.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("orchestrator.vision")

LOCK_SCRIPT = Path(__file__).resolve().parent.parent / "vision-tools" / "2_lock_camera_linux.sh"
FLUSH_FRAMES = 5  # V4L2 buffers a few stale frames; read past them


def lock_camera(index: int) -> None:
    """Pin exposure / white balance / gain on /dev/video<index>. Settings are volatile."""
    dev = f"/dev/video{index}"
    if shutil.which("v4l2-ctl") is None:
        log.warning("v4l2-ctl not installed — %s NOT locked (sudo apt install v4l-utils)", dev)
        return
    result = subprocess.run(["bash", str(LOCK_SCRIPT), dev], capture_output=True, text=True)
    for line in (result.stdout + result.stderr).strip().splitlines():
        log.info("lock %s: %s", dev, line)
    if result.returncode != 0:
        raise RuntimeError(f"camera lock failed on {dev}")


class Camera:
    """A V4L2 camera opened through OpenCV, with frame flushing on every grab."""

    def __init__(self, index: int, width: int, height: int):
        self.index = index
        self.width = width
        self.height = height
        self._cap = None

    def open(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera /dev/video{self.index} would not open")

    def grab(self):
        ok, frame = False, None
        for _ in range(FLUSH_FRAMES):
            ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"camera /dev/video{self.index} read failed")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def crop_roi(frame, roi):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi
    return frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def count_parts(frame, roi, min_area: int, dark_parts: bool = True) -> int:
    """Blob count inside `roi` of a BGR/gray frame. Parts must contrast with the table."""
    import cv2
    import numpy as np

    if frame is None:
        return 0
    crop = crop_roi(frame, roi)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    flag = cv2.THRESH_BINARY_INV if dark_parts else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    # Otsu on an EMPTY table splits noise into two halves; reject that case by
    # requiring the foreground to be a small fraction of the ROI
    if mask.mean() / 255.0 > 0.5:
        return 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) >= min_area)


class OnnxClassifier:
    """YOLOv8n-cls ONNX export: 224x224 RGB, /255, NCHW -> class scores."""

    def __init__(self, model_path: str, labels: list[str], size: int = 224):
        self.model_path = model_path
        self.labels = labels
        self.size = size
        self._sess = None

    def open(self) -> None:
        import onnxruntime as ort

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"classifier model not found: {self.model_path}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(self.model_path, providers=providers)
        self._input = self._sess.get_inputs()[0].name
        shape = self._sess.get_inputs()[0].shape
        if isinstance(shape[-1], int):
            self.size = shape[-1]
        n_out = self._sess.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(self.labels):
            raise ValueError(f"model has {n_out} classes but classifier.labels lists {len(self.labels)}")
        log.info("ONNX classifier %s (%s)", self.model_path, self._sess.get_providers()[0])

    def classify(self, frame) -> tuple[str, float]:
        import cv2
        import numpy as np

        img = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
        out = self._sess.run(None, {self._input: x})[0][0]
        probs = out if (out.min() >= 0 and abs(out.sum() - 1) < 1e-3) else _softmax(out)
        i = int(np.argmax(probs))
        return self.labels[i], float(probs[i])

    def close(self) -> None:
        self._sess = None


def _softmax(v):
    import numpy as np

    e = np.exp(v - v.max())
    return e / e.sum()


class CatalogClassifier:
    """Reuses autosort/classification (measure in mm, snap to vex_parts.yaml).
    Needs a reference frame of the EMPTY cup — pass it to `set_background()` at INIT."""

    def __init__(self, catalog_cfg: dict, labels: list[str]):
        self.catalog_cfg = catalog_cfg
        self.labels = labels
        self._clf = None

    def open(self) -> None:
        from autosort.classification.catalog import Catalog
        from autosort.classification.classifier import Classifier
        from autosort.config import CameraCfg, ClassifierCfg

        cfg = ClassifierCfg(
            catalog=self.catalog_cfg.get("path", "autosort/classification/vex_parts.yaml"),
            labels=list(self.labels) + ["unknown"],
            px_per_mm=float(self.catalog_cfg.get("px_per_mm", 12.0)),
            max_distance=float(self.catalog_cfg.get("max_distance", 1.5)),
            min_confidence=float(self.catalog_cfg.get("min_confidence", 0.35)),
            settle_s=0.0,
        )
        self._clf = Classifier(cfg, CameraCfg(index=-1))
        self._clf.catalog = Catalog.load(cfg.catalog)

    def set_background(self, empty_cup_frame) -> None:
        self._clf.camera.background = empty_cup_frame

    def classify(self, frame) -> tuple[str, float]:
        if self._clf.camera.background is None:
            raise RuntimeError("CatalogClassifier needs set_background(empty cup frame) first")
        label, margin = self._clf.identify(frame)
        if label == "unknown":
            return "REJECT", 0.0
        return label, min(1.0, margin)

    def close(self) -> None:
        self._clf = None
