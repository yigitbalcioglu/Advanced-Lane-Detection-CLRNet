import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


def heading_angle_from_centerline(center_coeffs: np.ndarray, y_near: float, y_far: float) -> float:
    """Estimate heading in a vehicle-like frame.

    Image coordinates use +y downward, while vehicle forward is upward in image.
    We therefore build heading from two points and use forward distance as
    (y_near - y_far) so rightward drift gives positive heading.
    """
    if center_coeffs is None or len(center_coeffs) < 2:
        return 0.0

    if y_far >= y_near:
        y_far = y_near - 1.0

    x_near = float(np.polyval(center_coeffs, y_near))
    x_far = float(np.polyval(center_coeffs, y_far))
    dx = x_far - x_near
    dy_forward = y_near - y_far
    if abs(dy_forward) < 1e-6:
        return 0.0
    return math.atan2(dx, dy_forward)


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


def pick_left_right_lanes(
    lanes_xy: List[List[Tuple[int, int]]],
    width: int,
    y_ref: int,
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    if not lanes_xy:
        return None, None

    cx = width / 2.0
    left = None
    right = None
    left_dist = float("inf")
    right_dist = float("inf")

    lane_infos: List[Tuple[List[Tuple[int, int]], float]] = []

    for lane in lanes_xy:
        coeffs = fit_lane_poly(lane)
        if coeffs is None:
            continue

        x_ref = float(np.polyval(coeffs, y_ref))
        lane_infos.append((lane, x_ref))
        d = x_ref - cx
        if d <= 0 and abs(d) < left_dist:
            left_dist = abs(d)
            left = lane
        if d > 0 and abs(d) < right_dist:
            right_dist = abs(d)
            right = lane

    if not lane_infos:
        return None, None

    left_candidates = [(lane, x_ref) for lane, x_ref in lane_infos if x_ref < cx]
    right_candidates = [(lane, x_ref) for lane, x_ref in lane_infos if x_ref > cx]

    if left_candidates and right_candidates:
        expected_w = width * 0.30
        min_w = width * 0.18
        max_w = width * 0.62

        best_score = float("inf")
        best_pair: Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = None
        for l_lane, lx in left_candidates:
            for r_lane, rx in right_candidates:
                lane_w = rx - lx
                if lane_w < min_w or lane_w > max_w:
                    continue

                center_x = (lx + rx) * 0.5
                center_cost = abs(center_x - cx)
                width_cost = abs(lane_w - expected_w)
                score = center_cost + 0.35 * width_cost

                if score < best_score:
                    best_score = score
                    best_pair = (l_lane, r_lane)

        if best_pair is not None:
            return best_pair

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
            "center_x_px": float("nan"),
            "steer_deg": 0.0,
            "direction": "unknown",
        }

    x_ref = float(np.polyval(center, y_ref))
    lane_px = max(1.0, lane_width_m / max(1.0, width * 0.4))
    px_to_m = lane_width_m / max(1.0, width * 0.4)

    cte_px = x_ref - (width / 2.0)
    cte_m = cte_px * px_to_m

    # Use two-point heading in vehicle-forward frame for more stable direction sign.
    y_far = float(height * 0.60)
    theta = heading_angle_from_centerline(center, float(y_ref), y_far)
    curvature = curvature_radius(center, y_ref)

    stanley = stanley or StanleyController()
    steer = stanley.steering(theta, cte_m, speed_mps)

    return {
        "has_center": 1.0,
        "curvature_m": float(curvature),
        "heading_error_deg": math.degrees(theta),
        "cross_track_error_m": float(cte_m),
        "center_x_px": float(x_ref),
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


class LaneSafetyMonitor:
    """Lane Departure Warning (LDW) and lane change detection.

    LDW triggers when cross-track error stays above threshold for several
    consecutive frames.  Lane change is flagged when the estimated centerline
    position jumps laterally by more than a pixel threshold in a single frame.
    """

    def __init__(
        self,
        ldw_cte_threshold_m: float = 0.45,
        ldw_frames: int = 8,
        lc_px_threshold: float = 55.0,
        lc_display_frames: int = 45,
        lc_cooldown_frames: int = 30,
    ):
        self._cte_buf: List[float] = []
        self._ldw_n = ldw_frames
        self._ldw_thr = ldw_cte_threshold_m

        self._prev_cx: Optional[float] = None
        self._lc_px_thr = lc_px_threshold
        self._lc_label = ""
        self._lc_ttl = 0
        self._lc_display_frames = lc_display_frames
        self._lc_cooldown = 0
        self._lc_cooldown_frames = lc_cooldown_frames

    def update(self, cte_m: float, center_x_px: Optional[float]) -> Dict[str, Any]:
        # --- Lane Departure Warning ---
        self._cte_buf.append(cte_m)
        if len(self._cte_buf) > self._ldw_n:
            self._cte_buf.pop(0)

        ldw_active = (
            len(self._cte_buf) == self._ldw_n
            and all(abs(c) >= self._ldw_thr for c in self._cte_buf)
        )
        ldw_direction = ("RIGHT" if cte_m > 0 else "LEFT") if ldw_active else ""

        # --- Lane Change Detection ---
        if self._lc_ttl > 0:
            self._lc_ttl -= 1
        if self._lc_cooldown > 0:
            self._lc_cooldown -= 1

        if center_x_px is not None and not math.isnan(center_x_px):
            if self._prev_cx is not None and self._lc_cooldown == 0:
                shift = center_x_px - self._prev_cx
                if abs(shift) >= self._lc_px_thr:
                    self._lc_label = "LANE CHANGE RIGHT" if shift > 0 else "LANE CHANGE LEFT"
                    self._lc_ttl = self._lc_display_frames
                    self._lc_cooldown = self._lc_cooldown_frames
            self._prev_cx = center_x_px

        return {
            "ldw_active": ldw_active,
            "ldw_direction": ldw_direction,
            "lane_change": self._lc_label if self._lc_ttl > 0 else "",
        }


class LaneTrack:
    """Internal state for one tracked lane."""

    def __init__(self, track_id: int, x_smooth: float):
        self.track_id: int = track_id
        self.x_smooth: float = x_smooth
        self.missing: int = 0


class LaneTracker:
    """
    Assigns stable numeric IDs to detected lanes across frames using
    greedy nearest-neighbour matching on x-position at y_ref.

    Matching threshold : 80 px
    Smoothing          : EMA with alpha=0.7 (new observation weight)
    Track removal      : after 8 consecutive missing frames
    """

    MATCH_THRESHOLD: float = 80.0
    EMA_ALPHA: float = 0.7
    MAX_MISSING: int = 8

    def __init__(self):
        self._tracks: List[LaneTrack] = []
        self._next_id: int = 0

    def update(
        self,
        lane_polylines: List[List[Tuple[int, int]]],
        width: int,
        y_ref: int,
    ) -> List[int]:
        """Match incoming polylines to existing tracks. Returns track IDs."""
        det_xs: List[float] = []
        for pts in lane_polylines:
            coeffs = fit_lane_poly(pts)
            if coeffs is not None:
                det_xs.append(float(np.polyval(coeffs, y_ref)))
            else:
                det_xs.append(float(pts[-1][0]) if pts else float(width / 2.0))

        n_det = len(det_xs)
        n_trk = len(self._tracks)
        matched_det: set = set()
        matched_trk: set = set()

        if n_det > 0 and n_trk > 0:
            costs = []
            for di, dx in enumerate(det_xs):
                for ti, trk in enumerate(self._tracks):
                    dist = abs(dx - trk.x_smooth)
                    if dist < self.MATCH_THRESHOLD:
                        costs.append((dist, di, ti))
            costs.sort(key=lambda c: c[0])

            for dist, di, ti in costs:
                if di in matched_det or ti in matched_trk:
                    continue
                self._tracks[ti].x_smooth = (
                    self.EMA_ALPHA * det_xs[di]
                    + (1.0 - self.EMA_ALPHA) * self._tracks[ti].x_smooth
                )
                self._tracks[ti].missing = 0
                matched_det.add(di)
                matched_trk.add(ti)

        for di in range(n_det):
            if di not in matched_det:
                new_track = LaneTrack(track_id=self._next_id, x_smooth=det_xs[di])
                self._next_id += 1
                self._tracks.append(new_track)
                matched_trk.add(len(self._tracks) - 1)
                matched_det.add(di)

        for ti in range(n_trk):
            if ti not in matched_trk:
                self._tracks[ti].missing += 1
        self._tracks = [t for t in self._tracks if t.missing <= self.MAX_MISSING]

        track_ids: List[int] = []
        for di, dx in enumerate(det_xs):
            best_trk = min(self._tracks, key=lambda t: abs(t.x_smooth - dx), default=None)
            track_ids.append(best_trk.track_id if best_trk is not None else -1)

        return track_ids

    def semantic_label(
        self,
        track_ids: List[int],
        lane_polylines: List[List[Tuple[int, int]]],
        width: int,
        y_ref: int,
    ) -> List[str]:
        """Return semantic label for each lane: EGO-L, EGO-R, L2, R2, etc."""
        left, right = pick_left_right_lanes(list(lane_polylines), width, y_ref)

        left_idx: Optional[int] = None
        right_idx: Optional[int] = None
        for i, pts in enumerate(lane_polylines):
            if left is not None and pts is left:
                left_idx = i
            if right is not None and pts is right:
                right_idx = i

        xs_indexed: List[Tuple[float, int]] = []
        for i, pts in enumerate(lane_polylines):
            coeffs = fit_lane_poly(pts)
            xr = float(np.polyval(coeffs, y_ref)) if coeffs is not None else float(pts[-1][0])
            xs_indexed.append((xr, i))
        xs_indexed.sort(key=lambda t: t[0])
        sorted_indices = [idx for _, idx in xs_indexed]

        ego_l_rank = sorted_indices.index(left_idx) if left_idx is not None else None
        ego_r_rank = sorted_indices.index(right_idx) if right_idx is not None else None

        label_map: dict = {}
        if ego_l_rank is not None:
            label_map[sorted_indices[ego_l_rank]] = "EGO-L"
        if ego_r_rank is not None:
            label_map[sorted_indices[ego_r_rank]] = "EGO-R"

        left_anchor = ego_l_rank if ego_l_rank is not None else (ego_r_rank if ego_r_rank is not None else 0)
        for n, rank in enumerate(range(left_anchor - 1, -1, -1), start=2):
            label_map[sorted_indices[rank]] = f"L{n}"

        right_anchor = ego_r_rank if ego_r_rank is not None else (ego_l_rank if ego_l_rank is not None else len(sorted_indices) - 1)
        for n, rank in enumerate(range(right_anchor + 1, len(sorted_indices)), start=2):
            label_map[sorted_indices[rank]] = f"R{n}"

        return [label_map.get(i, "?") for i in range(len(lane_polylines))]
