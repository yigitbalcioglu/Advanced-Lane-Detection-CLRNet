import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from scipy.interpolate import InterpolatedUnivariateSpline

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except Exception:  # pragma: no cover - optional runtime dependency
    AutoImageProcessor = None
    AutoModelForDepthEstimation = None

from clrnet.ops import nms
from clrnet.utils.advanced_lane_pipeline import (
    LaneSafetyMonitor,
    LaneTracker,
    PolyKalman,
    StanleyController,
    bev_matrix,
    default_bev_config,
    frame_metrics,
    fit_lane_poly,
    pick_left_right_lanes,
    sliding_window_angle_from_bev,
    warp_points,
)


@dataclass(frozen=True)
class OnnxPipelineConfig:
    model_path: str
    cut_height: int = 400
    cut_bottom: int = 0
    conf_threshold: float = 0.4
    nms_threshold: int = 50
    max_lanes: int = 5
    img_w: int = 1280
    img_h: int = 720
    input_width: int = 800
    input_height: int = 320
    force_gpu: bool = True
    use_bev: bool = True
    speed_mps: float = 12.0
    lane_width_m: float = 3.5
    show_dashboard: bool = True
    enable_yolo: bool = False
    yolo_model_path: str = "./weights/yolov8m.pt"
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    yolo_input_size: int = 960
    enable_depth: bool = False
    depth_model_path: str = "./weights/depth-anything-v2-small-hf"
    depth_overlay_alpha: float = 0.35
    depth_every_n_frames: int = 1
    enable_drivable_seg: bool = True
    enable_relative_speed: bool = True
    relative_speed_scale: float = 30.0


@dataclass(frozen=True)
class DetectedObject:
    label: str
    conf: float
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    label: str
    conf: float
    bbox: Tuple[int, int, int, int]
    hits: int
    age: int
    dh_dt: float


@dataclass(frozen=True)
class FCWInfo:
    active: bool
    level: str
    ttc_s: float
    lead_track_id: Optional[int]
    lead_bbox: Optional[Tuple[int, int, int, int]]
    risk: float


@dataclass(frozen=True)
class RelativeSpeedInfo:
    available: bool
    lead_track_id: Optional[int]
    rel_speed_mps: float
    closing: bool


@dataclass(frozen=True)
class DrivableAreaInfo:
    polygon: Optional[np.ndarray]
    ego_zone: str


class SimpleObjectTracker:
    def __init__(self, iou_match_threshold: float = 0.3, max_missed: int = 12):
        self.iou_match_threshold = float(iou_match_threshold)
        self.max_missed = int(max_missed)
        self.next_id = 1
        self.tracks: Dict[int, dict] = {}

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return float(inter / float(area_a + area_b - inter))

    def update(self, detections: Sequence[DetectedObject], frame_h: int, fps: float) -> List[TrackedObject]:
        for tr in self.tracks.values():
            tr["missed"] += 1
            tr["age"] += 1

        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))

        candidates: List[Tuple[float, int, int]] = []
        for tid, tr in self.tracks.items():
            for di, det in enumerate(detections):
                if tr["label"] != det.label:
                    continue
                score = self._iou(tr["bbox"], det.bbox)
                if score >= self.iou_match_threshold:
                    candidates.append((score, tid, di))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, tid, di in candidates:
            if tid not in unmatched_tracks or di not in unmatched_dets:
                continue
            det = detections[di]
            tr = self.tracks[tid]

            bh = max(1.0, float(det.bbox[3] - det.bbox[1]))
            bh_norm = bh / max(1.0, float(frame_h))
            prev_bh_norm = float(tr.get("bh_norm", bh_norm))
            dh_dt = (bh_norm - prev_bh_norm) * float(fps)

            tr["bbox"] = det.bbox
            tr["conf"] = det.conf
            tr["missed"] = 0
            tr["hits"] += 1
            tr["bh_norm"] = bh_norm
            tr["dh_dt"] = 0.7 * float(tr.get("dh_dt", 0.0)) + 0.3 * dh_dt

            unmatched_tracks.remove(tid)
            unmatched_dets.remove(di)

        for di in list(unmatched_dets):
            det = detections[di]
            bh = max(1.0, float(det.bbox[3] - det.bbox[1]))
            bh_norm = bh / max(1.0, float(frame_h))
            self.tracks[self.next_id] = {
                "label": det.label,
                "bbox": det.bbox,
                "conf": det.conf,
                "hits": 1,
                "age": 1,
                "missed": 0,
                "bh_norm": bh_norm,
                "dh_dt": 0.0,
            }
            self.next_id += 1

        stale = [tid for tid, tr in self.tracks.items() if tr["missed"] > self.max_missed]
        for tid in stale:
            del self.tracks[tid]

        out: List[TrackedObject] = []
        for tid, tr in self.tracks.items():
            if tr["missed"] > 0:
                continue
            out.append(
                TrackedObject(
                    track_id=int(tid),
                    label=str(tr["label"]),
                    conf=float(tr["conf"]),
                    bbox=tuple(tr["bbox"]),
                    hits=int(tr["hits"]),
                    age=int(tr["age"]),
                    dh_dt=float(tr.get("dh_dt", 0.0)),
                )
            )

        out.sort(key=lambda t: t.track_id)
        return out


COLORS: Sequence[Tuple[int, int, int]] = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
)


class Lane:
    def __init__(self, points: np.ndarray, invalid_value: float = -2.0, metadata: Optional[dict] = None):
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01
        self.metadata = metadata or {}


class CLRNetOnnxPipeline:
    def __init__(self, cfg: OnnxPipelineConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ort_session = self._build_session(cfg)

        self.n_offsets = 72
        self.n_strips = 71
        self.prior_ys = np.linspace(1, 0, self.n_offsets)

        self.kalman = PolyKalman()
        self.stanley = StanleyController(k_gain=0.8)
        self.safety = LaneSafetyMonitor()
        self.tracker = LaneTracker()
        self.yolo_model = self._build_yolo_detector(cfg) if cfg.enable_yolo else None
        self.depth_processor, self.depth_model = self._build_depth_estimator(cfg) if cfg.enable_depth else (None, None)
        self.object_tracker = SimpleObjectTracker(iou_match_threshold=0.28, max_missed=12)

        self._yolo_id_to_label = {
            0: "yaya",
            2: "arac",
            3: "arac",
            5: "arac",
            7: "arac",
            11: "levha",
        }

    def _build_session(self, cfg: OnnxPipelineConfig) -> onnxruntime.InferenceSession:
        available = onnxruntime.get_available_providers()
        if cfg.force_gpu:
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("CUDAExecutionProvider bulunamadi. onnxruntime-gpu gerekli.")
            if self.device.type != "cuda":
                raise RuntimeError("torch.cuda kullanilamiyor. GPU aktif degil.")
            providers = ["CUDAExecutionProvider"]
        else:
            providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]

        session = onnxruntime.InferenceSession(cfg.model_path, providers=providers)
        print(f"ONNX providers: {session.get_providers()}")
        print(f"Torch device: {self.device}")
        return session

    def _build_yolo_detector(self, cfg: OnnxPipelineConfig):
        if YOLO is None:
            raise RuntimeError("YOLO aktif ancak ultralytics kurulu degil. 'pip install ultralytics' calistirin.")
        if not os.path.exists(cfg.yolo_model_path):
            raise FileNotFoundError(f"YOLO agirlik dosyasi bulunamadi: {cfg.yolo_model_path}")

        model = YOLO(cfg.yolo_model_path)
        print(f"YOLO model yuku: {cfg.yolo_model_path}")
        return model

    def _build_depth_estimator(self, cfg: OnnxPipelineConfig):
        if AutoImageProcessor is None or AutoModelForDepthEstimation is None:
            raise RuntimeError(
                "Depth Anything aktif ancak transformers kurulu degil. "
                "'pip install transformers huggingface_hub' calistirin."
            )
        if not os.path.exists(cfg.depth_model_path):
            raise FileNotFoundError(f"Depth model yolu bulunamadi: {cfg.depth_model_path}")

        processor = AutoImageProcessor.from_pretrained(cfg.depth_model_path)
        model = AutoModelForDepthEstimation.from_pretrained(cfg.depth_model_path)
        model = model.to(self.device)
        model.eval()
        print(f"Depth model yuku: {cfg.depth_model_path}")
        return processor, model

    @staticmethod
    def softmax(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def predictions_to_lanes(self, predictions: np.ndarray) -> List[Lane]:
        lanes: List[Lane] = []
        valid_h = self.cfg.img_h - self.cfg.cut_height - self.cfg.cut_bottom
        if valid_h <= 0:
            return lanes
        for lane in predictions:
            lane_xs = lane[6:]
            start = min(max(0, int(round(lane[2].item() * self.n_strips))), self.n_strips)
            length = int(round(lane[5].item()))
            end = min(start + length - 1, len(self.prior_ys) - 1)

            mask = ~((((lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0))[::-1].cumprod()[::-1]).astype(bool))
            lane_xs[end + 1 :] = -2
            lane_xs[:start][mask] = -2

            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = np.asarray(lane_xs, dtype=np.float64)
            lane_xs = np.flip(lane_xs, axis=0)
            lane_ys = np.flip(lane_ys, axis=0)
            lane_ys = (lane_ys * valid_h + self.cfg.cut_height) / self.cfg.img_h

            if len(lane_xs) <= 1:
                continue

            points = np.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1).squeeze(2)
            lanes.append(Lane(points=points, metadata={"conf": lane[1]}))

        return lanes

    def decode_output(self, output: np.ndarray) -> List[Lane]:
        decoded: List[List[Lane]] = []
        for predictions in output:
            scores = self.softmax(predictions[:, :2], 1)[:, 1]
            keep_indices = scores >= self.cfg.conf_threshold
            predictions = predictions[keep_indices]
            scores = scores[keep_indices]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            nms_predictions = np.concatenate([predictions[..., :4], predictions[..., 5:]], axis=-1)
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[..., 5:] = nms_predictions[..., 5:] * (self.cfg.img_w - 1)

            keep, num_to_keep, _ = nms(
                torch.tensor(nms_predictions).to(self.device),
                torch.tensor(scores).to(self.device),
                overlap=self.cfg.nms_threshold,
                top_k=self.cfg.max_lanes,
            )
            keep = keep[:num_to_keep].cpu().numpy()
            predictions = predictions[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = np.round(predictions[:, 5] * self.n_strips)
            decoded.append(self.predictions_to_lanes(predictions))

        return decoded[0] if decoded else []

    def infer_lanes(self, frame: np.ndarray) -> List[Lane]:
        if frame is None or frame.size == 0:
            return []
        h = frame.shape[0]
        top = self.cfg.cut_height
        bottom = self.cfg.cut_bottom
        if top < 0 or bottom < 0:
            return []
        if top + bottom >= h - 1:
            return []

        y1 = top
        y2 = h - bottom if bottom > 0 else h
        img = frame[y1:y2, :, :]
        img = cv2.resize(img, (self.cfg.input_width, self.cfg.input_height), cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(np.float32(img[:, :, :, np.newaxis]), (3, 2, 0, 1))

        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return self.decode_output(ort_outs[0])

    def infer_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        if self.yolo_model is None:
            return []

        # Fail-fast behavior keeps errors visible instead of silently dropping detections.
        try:
            results = self.yolo_model.predict(
                source=frame,
                conf=self.cfg.yolo_conf_threshold,
                iou=self.cfg.yolo_iou_threshold,
                imgsz=self.cfg.yolo_input_size,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"YOLO tahmin hatasi: {exc}") from exc

        if not results:
            return []

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        classes = r0.boxes.cls.detach().cpu().numpy().astype(int)
        confs = r0.boxes.conf.detach().cpu().numpy()
        boxes = r0.boxes.xyxy.detach().cpu().numpy()

        h, w = frame.shape[:2]
        detections: List[DetectedObject] = []
        for cls_id, conf, box in zip(classes, confs, boxes):
            label = self._yolo_id_to_label.get(int(cls_id))
            if label is None:
                continue
            x1, y1, x2, y2 = box
            detections.append(
                DetectedObject(
                    label=label,
                    conf=float(conf),
                    bbox=(
                        int(np.clip(x1, 0, w - 1)),
                        int(np.clip(y1, 0, h - 1)),
                        int(np.clip(x2, 0, w - 1)),
                        int(np.clip(y2, 0, h - 1)),
                    ),
                )
            )

        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections

    def infer_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.depth_model is None or self.depth_processor is None:
            return None

        # Fail fast if depth estimation cannot run.
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self.depth_processor(images=rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth

            depth = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1).squeeze(0)
            depth_np = depth.detach().cpu().numpy().astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Depth Anything tahmin hatasi: {exc}") from exc

        lo = float(np.percentile(depth_np, 2.0))
        hi = float(np.percentile(depth_np, 98.0))
        if hi <= lo:
            return None

        depth_norm = np.clip((depth_np - lo) / (hi - lo), 0.0, 1.0)
        return (depth_norm * 255.0).astype(np.uint8)

    @staticmethod
    def lanes_to_pixel_polylines(frame: np.ndarray, lanes: Sequence[Lane]) -> List[List[Tuple[int, int]]]:
        h, w = frame.shape[:2]
        polys: List[List[Tuple[int, int]]] = []

        for lane in lanes:
            pts: List[Tuple[int, int]] = []
            for x, y in lane.points:
                px = int(x * w)
                py = int(y * h)
                if 0 <= px < w and 0 <= py < h:
                    pts.append((px, py))
            if len(pts) >= 2:
                polys.append(pts)

        # Sort by x at a stable bottom reference line instead of first point.
        y_ref = int(h * 0.90)
        scored: List[Tuple[float, List[Tuple[int, int]]]] = []
        for pts in polys:
            coeffs = fit_lane_poly(pts)
            if coeffs is not None:
                x_ref = float(np.polyval(coeffs, y_ref))
            else:
                x_ref = float(pts[-1][0])
            scored.append((x_ref, pts))
        scored.sort(key=lambda item: item[0])
        polys = [p for _, p in scored]
        return polys

    @staticmethod
    def _lane_color_map(
        lane_polylines: Sequence[Sequence[Tuple[int, int]]],
        width: int,
        y_ref: int,
    ) -> List[Tuple[int, int, int]]:
        """Assign stable semantic colors: ego-left, ego-right, and outer lanes."""
        # BGR
        color_ego_left = (0, 255, 0)
        color_ego_right = (0, 0, 255)
        color_outer_left = (255, 255, 0)
        color_outer_right = (255, 200, 0)
        color_fallback = (200, 200, 200)

        left, right = pick_left_right_lanes(list(lane_polylines), width, y_ref)

        lane_infos = []
        for i, pts in enumerate(lane_polylines):
            coeffs = fit_lane_poly(list(pts))
            if coeffs is not None:
                x_ref = float(np.polyval(coeffs, y_ref))
            else:
                x_ref = float(pts[-1][0])
            lane_infos.append((i, pts, x_ref))

        lane_infos.sort(key=lambda item: item[2])

        colors = [color_fallback for _ in lane_polylines]
        if not lane_infos:
            return colors

        if left is not None or right is not None:
            for i, pts, _ in lane_infos:
                if left is not None and pts is left:
                    colors[i] = color_ego_left
                elif right is not None and pts is right:
                    colors[i] = color_ego_right
                else:
                    cx = width / 2.0
                    coeffs = fit_lane_poly(list(pts))
                    x_ref = float(np.polyval(coeffs, y_ref)) if coeffs is not None else float(pts[-1][0])
                    colors[i] = color_outer_left if x_ref < cx else color_outer_right
        else:
            # Fallback semantic split with no clear ego pair.
            cx = width / 2.0
            for i, pts, x_ref in lane_infos:
                colors[i] = color_outer_left if x_ref < cx else color_outer_right

        return colors

    @staticmethod
    def _lane_color_map_tracked(
        lane_polylines: Sequence[Sequence[Tuple[int, int]]],
        labels: Sequence[str],
    ) -> List[Tuple[int, int, int]]:
        """Assign semantic colors using pre-computed track labels."""
        color_map = {
            "EGO-L": (0, 255, 0),
            "EGO-R": (0, 0, 255),
        }
        colors: List[Tuple[int, int, int]] = []
        for label in labels:
            if label in color_map:
                colors.append(color_map[label])
            elif label.startswith("L"):
                colors.append((255, 255, 0))
            elif label.startswith("R"):
                colors.append((255, 200, 0))
            else:
                colors.append((200, 200, 200))
        return colors

    @staticmethod
    def _draw_lane_labels(
        vis: np.ndarray,
        lane_polylines: Sequence[Sequence[Tuple[int, int]]],
        labels: Sequence[str],
        colors: Sequence[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Draw a semantic label badge at the vertical midpoint of each lane."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        pad = 4

        for pts, label, color in zip(lane_polylines, labels, colors):
            if not pts:
                continue
            mid = pts[len(pts) // 2]
            tx, ty = mid
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            bg_color = tuple(max(0, int(c * 0.55)) for c in color)
            cv2.rectangle(vis, (tx - pad, ty - th - pad), (tx + tw + pad, ty + baseline + pad), bg_color, -1)
            cv2.putText(vis, label, (tx, ty), font, font_scale, (255, 255, 255), thickness)

        return vis

    @staticmethod
    def draw_lane_corridor(vis: np.ndarray, lane_polylines: Sequence[Sequence[Tuple[int, int]]], color=(0, 255, 0), alpha=0.22) -> np.ndarray:
        h, w = vis.shape[:2]
        y_ref = int(h * 0.90)
        left, right = pick_left_right_lanes(lane_polylines, w, y_ref)
        if left is None or right is None:
            return vis

        left_c = fit_lane_poly(left)
        right_c = fit_lane_poly(right)
        if left_c is None or right_c is None:
            return vis

        y_top = int(h * 0.45)
        y_bot = int(h * 0.95)
        ys = np.linspace(y_top, y_bot, 48)

        xl = np.polyval(left_c, ys)
        xr = np.polyval(right_c, ys)

        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []
        for xli, xri, yi in zip(xl, xr, ys):
            yi_int = int(yi)
            xl_int = int(np.clip(xli, 0, w - 1))
            xr_int = int(np.clip(xri, 0, w - 1))
            if xl_int >= xr_int:
                continue
            left_pts.append((xl_int, yi_int))
            right_pts.append((xr_int, yi_int))

        if len(left_pts) < 3 or len(right_pts) < 3:
            return vis

        polygon = np.array(left_pts + list(reversed(right_pts)), dtype=np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [polygon], color)
        return cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0)

    @staticmethod
    def draw_object_detections(vis: np.ndarray, detections: Sequence[TrackedObject]) -> np.ndarray:
        color_map = {
            "arac": (0, 200, 255),
            "yaya": (255, 120, 0),
            "levha": (0, 255, 180),
        }
        counts = {"arac": 0, "yaya": 0, "levha": 0}

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.label, (220, 220, 220))
            counts[det.label] = counts.get(det.label, 0) + 1

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            txt = f"{det.label}#{det.track_id} {det.conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top = max(0, y1 - th - baseline - 6)
            cv2.rectangle(vis, (x1, top), (x1 + tw + 8, y1), color, -1)
            cv2.putText(vis, txt, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        summary = f"OBJ arac:{counts.get('arac', 0)} yaya:{counts.get('yaya', 0)} levha:{counts.get('levha', 0)}"
        cv2.putText(vis, summary, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        return vis

    @staticmethod
    def _estimate_lane_center_x(
        lane_polylines: Sequence[Sequence[Tuple[int, int]]],
        width: int,
        height: int,
    ) -> Optional[float]:
        y_ref = int(height * 0.9)
        left, right = pick_left_right_lanes(lane_polylines, width, y_ref)
        if left is None or right is None:
            return None
        left_c = fit_lane_poly(list(left))
        right_c = fit_lane_poly(list(right))
        if left_c is None or right_c is None:
            return None
        xl = float(np.polyval(left_c, y_ref))
        xr = float(np.polyval(right_c, y_ref))
        if not np.isfinite(xl) or not np.isfinite(xr):
            return None
        return 0.5 * (xl + xr)

    @staticmethod
    def _evaluate_fcw(
        tracked_objects: Sequence[TrackedObject],
        lane_center_x: Optional[float],
        frame_w: int,
        frame_h: int,
    ) -> FCWInfo:
        cx_lane = float(frame_w) * 0.5 if lane_center_x is None else float(lane_center_x)

        best = None
        best_score = -1.0
        for obj in tracked_objects:
            if obj.label != "arac":
                continue
            x1, y1, x2, y2 = obj.bbox
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            if cy < 0.35 * frame_h:
                continue

            bh_norm = bh / max(1.0, float(frame_h))
            lane_score = 1.0 - min(1.0, abs(cx - cx_lane) / (0.32 * frame_w))
            lane_score = max(0.0, lane_score)
            proximity_score = float(np.clip((bh_norm - 0.08) / 0.42, 0.0, 1.0))
            shape_score = float(np.clip((bw / bh) / 2.0, 0.0, 1.0))
            score = 0.60 * lane_score + 0.35 * proximity_score + 0.05 * shape_score

            if score > best_score:
                best_score = score
                best = (obj, bh_norm, proximity_score, lane_score)

        if best is None:
            return FCWInfo(active=False, level="NONE", ttc_s=float("inf"), lead_track_id=None, lead_bbox=None, risk=0.0)

        obj, bh_norm, proximity_score, lane_score = best
        dh_dt = max(0.0, obj.dh_dt)
        if dh_dt <= 1e-4:
            ttc_s = float("inf")
        else:
            # TTC proxy based on box-height growth until a near-field size target.
            ttc_s = max(0.0, (0.60 - bh_norm) / dh_dt)

        closing_score = float(np.clip(dh_dt / 0.9, 0.0, 1.0))
        risk = float(np.clip(0.45 * proximity_score + 0.35 * closing_score + 0.20 * lane_score, 0.0, 1.0))

        level = "NONE"
        active = False
        if (ttc_s < 1.2) or (risk >= 0.80):
            level = "CRITICAL"
            active = True
        elif (ttc_s < 2.0) or (risk >= 0.60):
            level = "WARN"
            active = True

        return FCWInfo(
            active=active,
            level=level,
            ttc_s=float(ttc_s),
            lead_track_id=int(obj.track_id),
            lead_bbox=obj.bbox,
            risk=risk,
        )

    @staticmethod
    def _draw_fcw_overlay(vis: np.ndarray, fcw_info: FCWInfo) -> np.ndarray:
        if not fcw_info.active:
            return vis

        h, w = vis.shape[:2]
        color = (0, 165, 255) if fcw_info.level == "WARN" else (0, 0, 255)
        label = f"FCW {fcw_info.level}"
        ttc_txt = "inf" if not np.isfinite(fcw_info.ttc_s) else f"{fcw_info.ttc_s:.2f}s"
        msg = f"{label} | TTC: {ttc_txt}"

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), color, -1)
        vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0)
        cv2.putText(vis, msg, (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        if fcw_info.lead_bbox is not None:
            x1, y1, x2, y2 = fcw_info.lead_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        return vis

    @staticmethod
    def _estimate_relative_speed(
        tracked_objects: Sequence[TrackedObject],
        fcw_info: FCWInfo,
        speed_scale: float,
    ) -> RelativeSpeedInfo:
        if fcw_info.lead_track_id is None:
            return RelativeSpeedInfo(available=False, lead_track_id=None, rel_speed_mps=0.0, closing=False)

        lead = None
        for obj in tracked_objects:
            if obj.track_id == fcw_info.lead_track_id:
                lead = obj
                break

        if lead is None:
            return RelativeSpeedInfo(available=False, lead_track_id=None, rel_speed_mps=0.0, closing=False)

        rel_speed = float(lead.dh_dt) * float(speed_scale)
        return RelativeSpeedInfo(
            available=True,
            lead_track_id=int(lead.track_id),
            rel_speed_mps=rel_speed,
            closing=rel_speed > 0.15,
        )

    @staticmethod
    def _build_drivable_polygon(
        lane_polylines: Sequence[Sequence[Tuple[int, int]]],
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        y_top = int(frame_h * 0.45)
        y_bot = int(frame_h * 0.98)
        ys = np.linspace(y_top, y_bot, 40)
        y_ref = int(frame_h * 0.9)
        left, right = pick_left_right_lanes(lane_polylines, frame_w, y_ref)

        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []
        if left is not None and right is not None:
            left_c = fit_lane_poly(list(left))
            right_c = fit_lane_poly(list(right))
            if left_c is not None and right_c is not None:
                for yi in ys:
                    xi_l = int(np.clip(np.polyval(left_c, yi), 0, frame_w - 1))
                    xi_r = int(np.clip(np.polyval(right_c, yi), 0, frame_w - 1))
                    if xi_l < xi_r:
                        left_pts.append((xi_l, int(yi)))
                        right_pts.append((xi_r, int(yi)))

        # Fallback trapezoid when lane pair is not reliable.
        if len(left_pts) < 6 or len(right_pts) < 6:
            top_half = int(frame_w * 0.11)
            bot_half = int(frame_w * 0.31)
            cx = frame_w // 2
            poly = np.array(
                [
                    (max(0, cx - top_half), y_top),
                    (min(frame_w - 1, cx + top_half), y_top),
                    (min(frame_w - 1, cx + bot_half), y_bot),
                    (max(0, cx - bot_half), y_bot),
                ],
                dtype=np.int32,
            )
            return poly

        return np.array(left_pts + list(reversed(right_pts)), dtype=np.int32)

    @staticmethod
    def _draw_drivable_overlay(vis: np.ndarray, drivable_info: DrivableAreaInfo) -> np.ndarray:
        poly = drivable_info.polygon
        if poly is None or len(poly) < 3:
            return vis

        h, w = vis.shape[:2]
        overlay = vis.copy()
        cv2.fillPoly(overlay, [poly], (40, 190, 40))

        # Side zones: practical sidewalk proxy in lower half.
        y_side = int(h * 0.55)
        side_poly_left = np.array([(0, y_side), (int(w * 0.2), y_side), (int(w * 0.2), h - 1), (0, h - 1)], dtype=np.int32)
        side_poly_right = np.array(
            [(int(w * 0.8), y_side), (w - 1, y_side), (w - 1, h - 1), (int(w * 0.8), h - 1)], dtype=np.int32
        )
        cv2.fillPoly(overlay, [side_poly_left], (220, 170, 50))
        cv2.fillPoly(overlay, [side_poly_right], (220, 170, 50))

        vis = cv2.addWeighted(overlay, 0.20, vis, 0.80, 0)
        cv2.polylines(vis, [poly], True, (50, 255, 50), 2)
        cv2.putText(vis, f"Drivable: {drivable_info.ego_zone}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        return vis

    @staticmethod
    def _estimate_ego_zone(drivable_poly: np.ndarray, frame_w: int, frame_h: int) -> str:
        if drivable_poly is None or len(drivable_poly) < 3:
            return "BILINMIYOR"

        ego_pt = (int(frame_w * 0.5), int(frame_h * 0.95))
        in_drivable = cv2.pointPolygonTest(drivable_poly, ego_pt, False) >= 0
        if in_drivable:
            return "YOL"

        if ego_pt[0] < int(frame_w * 0.2) or ego_pt[0] > int(frame_w * 0.8):
            return "KALDIRIM"
        return "SERIT DISI"

    @staticmethod
    def _draw_relative_speed(vis: np.ndarray, rel_speed: RelativeSpeedInfo) -> np.ndarray:
        if not rel_speed.available:
            cv2.putText(vis, "Lead RelSpeed: N/A", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 220), 2)
            return vis

        state = "YAKLASIYOR" if rel_speed.closing else "UZAKLASIYOR/SABIT"
        color = (0, 200, 255) if rel_speed.closing else (120, 220, 120)
        txt = f"Lead RelSpeed: {rel_speed.rel_speed_mps:+.2f} m/s ({state})"
        cv2.putText(vis, txt, (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        return vis

    def draw_depth_overlay(self, vis: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        alpha = float(np.clip(self.cfg.depth_overlay_alpha, 0.0, 0.9))
        blended = cv2.addWeighted(depth_color, alpha, vis, 1.0 - alpha, 0)

        # Keep the scene visible while still showing depth in a dedicated inset.
        h, w = vis.shape[:2]
        inset_w = max(220, w // 4)
        inset_h = max(120, h // 4)
        inset = cv2.resize(depth_color, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)

        x1 = w - inset_w - 16
        y1 = 16
        x2 = w - 16
        y2 = y1 + inset_h
        cv2.rectangle(blended, (x1 - 2, y1 - 28), (x2 + 2, y2 + 2), (20, 20, 20), -1)
        blended[y1:y2, x1:x2] = inset
        cv2.putText(blended, "Depth Anything", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
        return blended

    @staticmethod
    def draw_roi_polygon(vis: np.ndarray) -> np.ndarray:
        h, w = vis.shape[:2]
        cfg = default_bev_config()
        src = cfg.src_rel.copy()
        src[:, 0] *= w
        src[:, 1] *= h
        src = src.astype(np.int32)
        cv2.polylines(vis, [src], True, (0, 200, 255), 2)
        return src

    @staticmethod
    def _build_lane_mask(width: int, height: int, lane_polylines: Sequence[Sequence[Tuple[int, int]]]) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for pts in lane_polylines:
            if len(pts) < 2:
                continue
            arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [arr], False, 255, 8)
        return mask

    @staticmethod
    def _draw_safety_warnings(vis: np.ndarray, safety: dict) -> np.ndarray:
        """Draw LDW and lane change warning overlays on the frame."""
        h, w = vis.shape[:2]

        # --- Lane Departure Warning ---
        if safety["ldw_active"]:
            direction = safety["ldw_direction"]
            # Semi-transparent red border on departure side
            overlay = vis.copy()
            border_w = max(18, w // 20)
            if direction == "LEFT":
                cv2.rectangle(overlay, (0, 0), (border_w, h), (0, 0, 220), -1)
            else:
                cv2.rectangle(overlay, (w - border_w, 0), (w, h), (0, 0, 220), -1)
            vis = cv2.addWeighted(overlay, 0.40, vis, 0.60, 0)

            # Warning text centered at top
            warn_txt = f"! LDW: {direction} !"
            font_scale = max(0.9, w / 900)
            thickness = 2
            (tw, th), _ = cv2.getTextSize(warn_txt, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
            tx = (w - tw) // 2
            ty = th + 14
            cv2.rectangle(vis, (tx - 8, 4), (tx + tw + 8, ty + 8), (0, 0, 180), -1)
            cv2.putText(vis, warn_txt, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness)

        # --- Lane Change Detection ---
        lc = safety.get("lane_change", "")
        if lc:
            color = (0, 200, 255) if "RIGHT" in lc else (255, 180, 0)
            font_scale = max(0.8, w / 1100)
            thickness = 2
            (tw, th), _ = cv2.getTextSize(lc, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
            tx = (w - tw) // 2
            ty = h - 24
            cv2.rectangle(vis, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8), (30, 30, 30), -1)
            cv2.putText(vis, lc, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)

        return vis

    @staticmethod
    def draw_steering_widget(vis: np.ndarray, steer_deg: float) -> np.ndarray:
        """Draw a simple steering wheel HUD at the bottom-right of the frame."""
        h, w = vis.shape[:2]
        radius = max(34, min(h, w) // 12)
        margin = 24
        cx = w - margin - radius
        cy = h - margin - radius

        # Clamp for display stability when estimator spikes.
        disp_deg = float(np.clip(steer_deg, -45.0, 45.0))
        theta = np.deg2rad(disp_deg)

        wheel_color = (230, 230, 230)
        center_color = (70, 70, 70)
        needle_color = (0, 230, 255) if abs(disp_deg) < 25 else (0, 120, 255)

        overlay = vis.copy()
        cv2.circle(overlay, (cx, cy), radius + 12, (25, 25, 25), -1)
        vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

        cv2.circle(vis, (cx, cy), radius, wheel_color, 3)
        cv2.circle(vis, (cx, cy), radius // 4, center_color, -1)

        # Rotating spoke pair to make steering direction obvious.
        for spoke_offset in [0.0, np.pi * 0.66]:
            ang = theta + spoke_offset
            x2 = int(cx + np.cos(ang) * (radius - 5))
            y2 = int(cy + np.sin(ang) * (radius - 5))
            cv2.line(vis, (cx, cy), (x2, y2), wheel_color, 2)

        # Needle on top spoke for stronger response feedback.
        nx = int(cx + np.cos(theta - np.pi / 2.0) * (radius + 2))
        ny = int(cy + np.sin(theta - np.pi / 2.0) * (radius + 2))
        cv2.line(vis, (cx, cy), (nx, ny), needle_color, 3)

        txt = f"Steer {disp_deg:+.1f} deg"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = max(10, cx - tw // 2)
        ty = max(th + 8, cy - radius - 16)
        cv2.putText(vis, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2)
        return vis

    def _build_dashboard(
        self,
        frame_vis: np.ndarray,
        lane_polylines: Sequence[Sequence[Tuple[int, int]]],
        main_heading_deg: float,
    ) -> Tuple[np.ndarray, float]:
        h, w = frame_vis.shape[:2]
        panel_w = 420
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (22, 22, 22)

        lane_mask = self._build_lane_mask(w, h, lane_polylines)
        bevm, _ = bev_matrix(w, h)
        bev_lane = cv2.warpPerspective(lane_mask, bevm, (w, h))

        sw_angle_deg, sw_centers = sliding_window_angle_from_bev(bev_lane)

        bev_color = cv2.cvtColor(bev_lane, cv2.COLOR_GRAY2BGR)
        for x, y in sw_centers:
            cv2.circle(bev_color, (x, y), 4, (0, 255, 255), -1)

        top_h = h // 2
        bev_small = cv2.resize(bev_color, (panel_w - 20, top_h - 60), interpolation=cv2.INTER_NEAREST)
        panel[40 : 40 + bev_small.shape[0], 10 : 10 + bev_small.shape[1]] = bev_small
        cv2.putText(panel, "BEV Lane Mask", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        y0 = top_h + 40
        cv2.putText(panel, "Dashboard", (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, f"extra sw angle(deg): {sw_angle_deg:.2f}", (12, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 255, 120), 2)
        cv2.putText(panel, f"main heading(deg): {main_heading_deg:.2f}", (12, y0 + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 200, 255), 2)
        cv2.putText(panel, f"windows found: {len(sw_centers)}", (12, y0 + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2)

        return np.hstack([frame_vis, panel]), sw_angle_deg

    def annotate(
        self,
        frame: np.ndarray,
        lane_polylines: Sequence[Sequence[Tuple[int, int]]],
        object_detections: Optional[Sequence[TrackedObject]] = None,
        fcw_info: Optional[FCWInfo] = None,
        drivable_info: Optional[DrivableAreaInfo] = None,
        rel_speed_info: Optional[RelativeSpeedInfo] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        if depth_map is not None:
            vis = self.draw_depth_overlay(vis, depth_map)

        if self.cfg.use_bev:
            self.draw_roi_polygon(vis)
        vis = self.draw_lane_corridor(vis, lane_polylines)
        if drivable_info is not None and self.cfg.enable_drivable_seg:
            vis = self._draw_drivable_overlay(vis, drivable_info)

        bevm, _ = bev_matrix(w, h)
        # Always use BEV coordinates for metrics: removes perspective distortion
        # so heading/CTE/curvature reflect true road geometry regardless of display mode.
        metrics_input = [warp_points(pts, bevm) for pts in lane_polylines]

        metrics = frame_metrics(
            lanes_xy=metrics_input,
            width=w,
            height=h,
            speed_mps=self.cfg.speed_mps,
            lane_width_m=self.cfg.lane_width_m,
            kalman=self.kalman,
            stanley=self.stanley,
        )

        safety = self.safety.update(
            cte_m=metrics["cross_track_error_m"],
            center_x_px=metrics.get("center_x_px"),
        )
        vis = self._draw_safety_warnings(vis, safety)

        if fcw_info is not None:
            vis = self._draw_fcw_overlay(vis, fcw_info)

        y_ref_px = int(h * 0.90)
        track_ids = self.tracker.update(list(lane_polylines), w, y_ref_px)
        lane_labels = self.tracker.semantic_label(track_ids, list(lane_polylines), w, y_ref_px)
        lane_colors = self._lane_color_map_tracked(lane_polylines, lane_labels)
        for idx, pts in enumerate(lane_polylines):
            color = lane_colors[idx]
            for i in range(1, len(pts)):
                cv2.line(vis, pts[i - 1], pts[i], color, 3)
        vis = self._draw_lane_labels(vis, lane_polylines, lane_labels, lane_colors)

        mode = "BEV" if self.cfg.use_bev else "IMAGE"
        cv2.putText(vis, f"mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"cut_height: {self.cfg.cut_height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"cut_bottom: {self.cfg.cut_bottom}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"direction: {metrics['direction']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        curv_txt = f"curvature R: {metrics['curvature_m']:.1f}" if np.isfinite(metrics["curvature_m"]) else "curvature R: inf"
        cv2.putText(vis, curv_txt, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"heading err(deg): {metrics['heading_error_deg']:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"cte(m): {metrics['cross_track_error_m']:.3f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"stanley steer(deg): {metrics['steer_deg']:.2f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if object_detections:
            vis = self.draw_object_detections(vis, object_detections)

        # AEB-lite: unified risk score (0-100) from FCW + lane geometry + LDW state.
        fcw_risk = float(fcw_info.risk) if fcw_info is not None else 0.0
        cte_risk = float(np.clip(abs(metrics["cross_track_error_m"]) / 1.2, 0.0, 1.0))
        ldw_risk = 1.0 if safety.get("ldw_active", False) else 0.0
        aeb_risk = float(np.clip(0.60 * fcw_risk + 0.25 * cte_risk + 0.15 * ldw_risk, 0.0, 1.0))
        aeb_score = int(round(aeb_risk * 100.0))
        aeb_color = (80, 220, 80) if aeb_score < 40 else (0, 200, 255) if aeb_score < 70 else (0, 0, 255)
        cv2.putText(vis, f"AEB-lite risk: {aeb_score}/100", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.72, aeb_color, 2)
        if self.cfg.enable_relative_speed:
            vis = self._draw_relative_speed(vis, rel_speed_info or RelativeSpeedInfo(False, None, 0.0, False))

        if self.cfg.show_dashboard:
            vis, sw_angle = self._build_dashboard(vis, lane_polylines, metrics["heading_error_deg"])
            cv2.putText(vis, f"sw angle(deg): {sw_angle:.2f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        vis = self.draw_steering_widget(vis, metrics["steer_deg"])

        return vis

    def run_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video acilamadi: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None

        print(f"\nVideo: {video_path}")
        print(
            f"Boyut: {width}x{height} | FPS: {fps:.2f} | "
            f"cut_height={self.cfg.cut_height}, cut_bottom={self.cfg.cut_bottom}"
        )

        gui_enabled = show
        frame_count = 0
        cached_depth_map: Optional[np.ndarray] = None
        depth_stride = max(1, int(self.cfg.depth_every_n_frames))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lanes = self.infer_lanes(frame)
            lane_polys = self.lanes_to_pixel_polylines(frame, lanes)
            raw_detections = self.infer_objects(frame) if self.cfg.enable_yolo else []
            tracked_detections = self.object_tracker.update(raw_detections, frame_h=height, fps=fps)
            lane_center_x = self._estimate_lane_center_x(lane_polys, width, height)
            fcw_info = self._evaluate_fcw(tracked_detections, lane_center_x, width, height)
            rel_speed_info = self._estimate_relative_speed(
                tracked_detections,
                fcw_info,
                speed_scale=float(self.cfg.relative_speed_scale),
            )
            drivable_poly = self._build_drivable_polygon(lane_polys, width, height) if self.cfg.enable_drivable_seg else None
            ego_zone = self._estimate_ego_zone(drivable_poly, width, height) if drivable_poly is not None else "BILINMIYOR"
            drivable_info = DrivableAreaInfo(polygon=drivable_poly, ego_zone=ego_zone)
            if self.cfg.enable_depth:
                # Run depth model periodically and reuse last depth map in-between for speed.
                if (frame_count % depth_stride) == 0 or cached_depth_map is None:
                    cached_depth_map = self.infer_depth(frame)
                depth_map = cached_depth_map
            else:
                depth_map = None
            vis = self.annotate(
                frame,
                lane_polys,
                object_detections=tracked_detections,
                fcw_info=fcw_info,
                drivable_info=drivable_info,
                rel_speed_info=rel_speed_info,
                depth_map=depth_map,
            )

            if output_path and writer is None:
                out_h, out_w = vis.shape[:2]
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

            if writer is not None:
                writer.write(vis)

            if gui_enabled:
                try:
                    cv2.imshow("CLRNet Main Pipeline", vis)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                except cv2.error:
                    gui_enabled = False
                    print("OpenCV pencere acilamadi, terminal modunda devam.")

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
            if max_frames is not None and frame_count >= max_frames:
                break

        cap.release()
        if writer is not None:
            writer.release()
        if gui_enabled:
            cv2.destroyAllWindows()

        print(f"Tamamlandi. Islenen frame: {frame_count}")
        if output_path:
            print(f"Cikti video: {output_path}")


def resolve_video_path(video_name: str, videos_dir: str) -> str:
    if os.path.exists(video_name):
        return video_name

    direct = os.path.join(videos_dir, video_name)
    if os.path.exists(direct):
        return direct

    base, ext = os.path.splitext(video_name)
    if ext:
        raise FileNotFoundError(f"Video bulunamadi: {video_name}")

    for candidate_ext in [".mp4", ".mov", ".avi", ".mkv"]:
        candidate = os.path.join(videos_dir, base + candidate_ext)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Video bulunamadi: {video_name}")
