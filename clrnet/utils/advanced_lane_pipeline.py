import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def fit_lane_poly(points_xy: List[Tuple[int, int]], degree: int = 2) -> Optional[np.ndarray]:
    if points_xy is None or len(points_xy) < (degree + 1):
        return None
    pts = np.asarray(points_xy, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    try:
        coeffs = np.polyfit(y, x, degree)
    except Exception:
        return None
    return coeffs


def tangent_angle_rad(coeffs: np.ndarray, y_eval: float) -> float:
    # x(y) = Ay^2 + By + C -> dx/dy = 2Ay + B
    if coeffs is None or len(coeffs) < 2:
        return 0.0
    a = float(coeffs[0])
    b = float(coeffs[1])
    slope = (2.0 * a * y_eval) + b
    return math.atan(slope)


def curvature_radius(coeffs: np.ndarray, y_eval: float) -> float:
    # R = [1 + (2Ay + B)^2]^(3/2) / |2A|
    if coeffs is None or len(coeffs) < 3:
        return float("inf")
    a = float(coeffs[0])
    b = float(coeffs[1])
    denom = abs(2.0 * a)
    if denom < 1e-8:
        return float("inf")
    val = 1.0 + ((2.0 * a * y_eval + b) ** 2)
    return (val ** 1.5) / denom


@dataclass
class BEVConfig:
    src_rel: np.ndarray
    dst_rel: np.ndarray


def default_bev_config() -> BEVConfig:
    # Relative points for a generic front camera.
    src_rel = np.array(
        [[0.43, 0.62], [0.57, 0.62], [0.95, 0.98], [0.05, 0.98]], dtype=np.float32
    )
    dst_rel = np.array(
        [[0.30, 0.05], [0.70, 0.05], [0.70, 0.98], [0.30, 0.98]], dtype=np.float32
    )
    return BEVConfig(src_rel=src_rel, dst_rel=dst_rel)


def bev_matrix(width: int, height: int, cfg: Optional[BEVConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    cfg = cfg or default_bev_config()
    src = cfg.src_rel.copy()
    dst = cfg.dst_rel.copy()
    src[:, 0] *= width
    src[:, 1] *= height
    dst[:, 0] *= width
    dst[:, 1] *= height
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    return m, m_inv


def warp_points(points_xy: List[Tuple[int, int]], matrix: np.ndarray) -> List[Tuple[int, int]]:
    if not points_xy:
        return []
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
    return [(int(p[0]), int(p[1])) for p in out]


class PolyKalman:
    """3-state Kalman for quadratic polynomial coefficients [A, B, C]."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(3, 3)
        self.kf.transitionMatrix = np.eye(3, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(3, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(3, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 5e-2
        self.kf.errorCovPost = np.eye(3, dtype=np.float32)
        self.initialized = False

    def update(self, coeffs: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if coeffs is None:
            if not self.initialized:
                return None
            pred = self.kf.predict().reshape(-1)
            return pred.astype(np.float64)

        meas = np.asarray(coeffs, dtype=np.float32).reshape(3, 1)
        if not self.initialized:
            self.kf.statePost = meas
            self.initialized = True
            return meas.reshape(-1).astype(np.float64)

        self.kf.predict()
        est = self.kf.correct(meas).reshape(-1)
        return est.astype(np.float64)


class StanleyController:
    def __init__(self, k_gain: float = 0.8, max_steer_deg: float = 35.0):
        self.k_gain = k_gain
        self.max_steer_rad = math.radians(max_steer_deg)

    def steering(self, heading_error_rad: float, cross_track_error_m: float, speed_mps: float) -> float:
        v = max(speed_mps, 0.1)
        delta = heading_error_rad + math.atan((self.k_gain * cross_track_error_m) / v)
        delta = max(-self.max_steer_rad, min(self.max_steer_rad, delta))
        return math.degrees(delta)


def pick_left_right_lanes(lanes_xy: List[List[Tuple[int, int]]], width: int, y_ref: int) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    if not lanes_xy:
        return None, None

    cx = width // 2
    left = None
    right = None
    left_dist = float("inf")
    right_dist = float("inf")

    for lane in lanes_xy:
        coeffs = fit_lane_poly(lane)
        if coeffs is None:
            continue
        x_ref = np.polyval(coeffs, y_ref)
        d = x_ref - cx
        if d <= 0 and abs(d) < left_dist:
            left_dist = abs(d)
            left = lane
        if d > 0 and abs(d) < right_dist:
            right_dist = abs(d)
            right = lane

    return left, right


def centerline_from_lr(left: Optional[List[Tuple[int, int]]], right: Optional[List[Tuple[int, int]]], y_samples: np.ndarray) -> Optional[np.ndarray]:
    if left is None and right is None:
        return None

    left_c = fit_lane_poly(left) if left is not None else None
    right_c = fit_lane_poly(right) if right is not None else None

    if left_c is None and right_c is None:
        return None

    if left_c is None:
        return right_c
    if right_c is None:
        return left_c

    x_l = np.polyval(left_c, y_samples)
    x_r = np.polyval(right_c, y_samples)
    x_c = (x_l + x_r) / 2.0
    center_coeff = np.polyfit(y_samples, x_c, 2)
    return center_coeff


def lane_direction_label(theta_rad: float, deadzone_deg: float = 2.0) -> str:
    theta_deg = math.degrees(theta_rad)
    if theta_deg > deadzone_deg:
        return "right"
    if theta_deg < -deadzone_deg:
        return "left"
    return "straight"


def frame_metrics(
    lanes_xy: List[List[Tuple[int, int]]],
    width: int,
    height: int,
    speed_mps: float,
    lane_width_m: float = 3.5,
    kalman: Optional[PolyKalman] = None,
    stanley: Optional[StanleyController] = None,
) -> Dict[str, float]:
    y_ref = int(height * 0.90)
    y_samples = np.linspace(int(height * 0.45), y_ref, 32)

    left, right = pick_left_right_lanes(lanes_xy, width, y_ref)
    center = centerline_from_lr(left, right, y_samples)
    if kalman is not None:
        center = kalman.update(center)

    if center is None:
        return {
            "has_center": 0.0,
            "curvature_m": float("inf"),
            "heading_error_deg": 0.0,
            "cross_track_error_m": 0.0,
            "steer_deg": 0.0,
            "direction": "unknown",
        }

    x_ref = float(np.polyval(center, y_ref))
    lane_px = max(1.0, lane_width_m / max(1.0, width * 0.4))
    px_to_m = lane_width_m / max(1.0, width * 0.4)

    cte_px = x_ref - (width / 2.0)
    cte_m = cte_px * px_to_m

    theta = tangent_angle_rad(center, y_ref)
    curvature = curvature_radius(center, y_ref)

    stanley = stanley or StanleyController()
    steer = stanley.steering(theta, cte_m, speed_mps)

    return {
        "has_center": 1.0,
        "curvature_m": float(curvature),
        "heading_error_deg": math.degrees(theta),
        "cross_track_error_m": float(cte_m),
        "steer_deg": float(steer),
        "direction": lane_direction_label(theta),
    }


def sliding_window_angle_from_bev(
    bev_binary: np.ndarray,
    n_windows: int = 9,
    min_pixels: int = 40,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Estimate lane heading angle from BEV using a simple sliding-window center tracker.

    Returns:
        angle_deg: positive means right, negative means left, 0 means straight/unknown.
        centers: collected sliding window centers in image coordinates.
    """
    if bev_binary is None or bev_binary.size == 0:
        return 0.0, []

    if bev_binary.ndim == 3:
        gray = cv2.cvtColor(bev_binary, cv2.COLOR_BGR2GRAY)
    else:
        gray = bev_binary

    h, w = gray.shape[:2]
    nonzero = cv2.findNonZero(gray)
    if nonzero is None:
        return 0.0, []

    nonzero = nonzero.reshape(-1, 2)
    window_height = max(1, h // n_windows)

    histogram = np.sum(gray[h // 2 :, :], axis=0)
    base_x = int(np.argmax(histogram))
    margin = max(20, w // 12)

    centers: List[Tuple[int, int]] = []
    current_x = base_x

    for win in range(n_windows):
        y_low = h - (win + 1) * window_height
        y_high = h - win * window_height
        x_low = max(0, current_x - margin)
        x_high = min(w, current_x + margin)

        in_win = (
            (nonzero[:, 1] >= y_low)
            & (nonzero[:, 1] < y_high)
            & (nonzero[:, 0] >= x_low)
            & (nonzero[:, 0] < x_high)
        )
        pts = nonzero[in_win]
        if pts.shape[0] >= min_pixels:
            current_x = int(np.mean(pts[:, 0]))
            center_y = int((y_low + y_high) * 0.5)
            centers.append((current_x, center_y))

    if len(centers) < 2:
        return 0.0, centers

    ys = np.array([p[1] for p in centers], dtype=np.float32)
    xs = np.array([p[0] for p in centers], dtype=np.float32)
    try:
        coeff = np.polyfit(ys, xs, 1)
    except Exception:
        return 0.0, centers

    dx_dy = float(coeff[0])
    angle_deg = math.degrees(math.atan(dx_dy))
    return angle_deg, centers
