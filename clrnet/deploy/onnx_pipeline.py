import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime
import torch
from scipy.interpolate import InterpolatedUnivariateSpline

from clrnet.ops import nms
from clrnet.utils.advanced_lane_pipeline import (
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

    @staticmethod
    def softmax(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def predictions_to_lanes(self, predictions: np.ndarray) -> List[Lane]:
        lanes: List[Lane] = []
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
            lane_ys = (lane_ys * (self.cfg.img_h - self.cfg.cut_height) + self.cfg.cut_height) / self.cfg.img_h

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
        if self.cfg.cut_height >= frame.shape[0] - 1:
            return []

        img = frame[self.cfg.cut_height :, :, :]
        img = cv2.resize(img, (self.cfg.input_width, self.cfg.input_height), cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(np.float32(img[:, :, :, np.newaxis]), (3, 2, 0, 1))

        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return self.decode_output(ort_outs[0])

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

        polys.sort(key=lambda p: p[0][0])
        return polys

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

    def annotate(self, frame: np.ndarray, lane_polylines: Sequence[Sequence[Tuple[int, int]]]) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        self.draw_roi_polygon(vis)
        vis = self.draw_lane_corridor(vis, lane_polylines)

        bevm, _ = bev_matrix(w, h)
        metrics_input = lane_polylines
        if self.cfg.use_bev:
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

        for idx, pts in enumerate(lane_polylines):
            color = COLORS[idx % len(COLORS)]
            for i in range(1, len(pts)):
                cv2.line(vis, pts[i - 1], pts[i], color, 3)

        mode = "BEV" if self.cfg.use_bev else "IMAGE"
        cv2.putText(vis, f"mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"cut_height: {self.cfg.cut_height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"direction: {metrics['direction']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        curv_txt = f"curvature R: {metrics['curvature_m']:.1f}" if np.isfinite(metrics["curvature_m"]) else "curvature R: inf"
        cv2.putText(vis, curv_txt, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"heading err(deg): {metrics['heading_error_deg']:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"cte(m): {metrics['cross_track_error_m']:.3f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"stanley steer(deg): {metrics['steer_deg']:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.cfg.show_dashboard:
            vis, sw_angle = self._build_dashboard(vis, lane_polylines, metrics["heading_error_deg"])
            cv2.putText(vis, f"sw angle(deg): {sw_angle:.2f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
        print(f"Boyut: {width}x{height} | FPS: {fps:.2f} | cut_height={self.cfg.cut_height}")

        gui_enabled = show
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lanes = self.infer_lanes(frame)
            lane_polys = self.lanes_to_pixel_polylines(frame, lanes)
            vis = self.annotate(frame, lane_polys)

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
