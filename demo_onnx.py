import argparse
import os

from clrnet.deploy import CLRNetOnnxPipeline, OnnxPipelineConfig
from clrnet.deploy.onnx_pipeline import resolve_video_path


def parse_args():
    parser = argparse.ArgumentParser(description="CLRNet ONNX pipeline")
    parser.add_argument("video_name", help="Video adi veya tam path (orn: 2.mp4 veya videos/2.mp4)")
    parser.add_argument("--videos-dir", default="./videos", help="Video klasoru")
    parser.add_argument("--onnx", default="./tusimple_r18.onnx", help="ONNX model yolu")
    parser.add_argument("--conf-threshold", type=float, default=0.4)
    parser.add_argument("--output", default=None, help="Istege bagli cikti video yolu")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--enable-yolo", action="store_true", help="YOLOv8 nesne tespiti (arac/yaya/levha) ac")
    parser.add_argument("--yolo-weights", default="./weights/yolov8m.pt", help="YOLO agirlik dosyasi")
    parser.add_argument("--yolo-conf-threshold", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou-threshold", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--yolo-input-size", type=int, default=960, help="YOLO input boyutu")
    parser.add_argument("--enable-depth", action="store_true", help="Depth Anything derinlik tahmini ac")
    parser.add_argument(
        "--depth-model-path",
        default="./weights/depth-anything-v2-small-hf",
        help="Depth Anything model klasoru",
    )
    parser.add_argument(
        "--depth-overlay-alpha",
        type=float,
        default=0.35,
        help="Depth overlay saydamligi (0-0.9)",
    )
    parser.add_argument(
        "--depth-every-n-frames",
        type=int,
        default=1,
        help="Depth tahminini her N framede bir calistir (orn: 15)",
    )
    parser.add_argument("--disable-drivable-seg", action="store_true", help="Drivable area segmentation overlay kapat")
    parser.add_argument("--disable-relative-speed", action="store_true", help="Lead relative speed HUD kapat")
    parser.add_argument(
        "--relative-speed-scale",
        type=float,
        default=30.0,
        help="BBox buyume oranindan relatif hiz tahmini olcegi",
    )

    bev_group = parser.add_mutually_exclusive_group()
    bev_group.add_argument("--enable-bev", action="store_true", help="BEV donusumunu ac")
    bev_group.add_argument("--disable-bev", action="store_true", help="BEV donusumunu kapat")

    parser.add_argument("--speed-mps", type=float, default=12.0, help="Stanley icin hiz (m/s)")
    parser.add_argument("--lane-width-m", type=float, default=3.5, help="Standart serit genisligi")

    dashboard_group = parser.add_mutually_exclusive_group()
    dashboard_group.add_argument("--show-dashboard", action="store_true", help="Sag panel dashboardu ac")
    dashboard_group.add_argument("--no-dashboard", action="store_true", help="Sag panel dashboardu kapat")

    parser.add_argument("-ch", "--cut-height", type=int, default=480, help="Ustten kirpma miktari")
    parser.add_argument("-cb", "--cut-bottom", type=int, default=0, help="Alttan kirpma miktari")
    parser.add_argument(
        "--cut-heights",
        nargs="+",
        type=int,
        default=None,
        help="Birden fazla cut_height degeri ile ardisik test (orn: --cut-heights 220 240 260)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Sweep modunda her cut_height sonucu icin klasor (orn: outputs/cut_sweep)",
    )
    return parser.parse_args()


def build_output_path(video_path: str, output_path: str, output_dir: str, cut_height: int, is_sweep: bool):
    if not is_sweep:
        return output_path

    if output_path:
        base, ext = os.path.splitext(output_path)
        ext = ext or ".mp4"
        return f"{base}_ch{cut_height}{ext}"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(output_dir, f"{video_stem}_ch{cut_height}.mp4")

    return None


def main():
    args = parse_args()
    video_path = resolve_video_path(args.video_name, args.videos_dir)

    cut_heights = args.cut_heights if args.cut_heights else [args.cut_height]
    cut_heights = list(dict.fromkeys(cut_heights))
    if any(ch < 0 for ch in cut_heights):
        raise ValueError("cut_height negatif olamaz")
    if args.cut_bottom < 0:
        raise ValueError("cut_bottom negatif olamaz")

    is_sweep = len(cut_heights) > 1
    if is_sweep:
        print(f"Cut height sweep: {cut_heights}")

    # Default policy: BEV/dashboard are hidden unless explicitly enabled.
    use_bev = bool(args.enable_bev)
    show_dashboard = bool(args.show_dashboard)

    for cut_height in cut_heights:
        cfg = OnnxPipelineConfig(
            model_path=args.onnx,
            cut_height=cut_height,
            cut_bottom=args.cut_bottom,
            conf_threshold=args.conf_threshold,
            force_gpu=(not args.allow_cpu),
            use_bev=use_bev,
            speed_mps=args.speed_mps,
            lane_width_m=args.lane_width_m,
            show_dashboard=show_dashboard,
            enable_yolo=bool(args.enable_yolo),
            yolo_model_path=args.yolo_weights,
            yolo_conf_threshold=args.yolo_conf_threshold,
            yolo_iou_threshold=args.yolo_iou_threshold,
            yolo_input_size=args.yolo_input_size,
            enable_depth=bool(args.enable_depth),
            depth_model_path=args.depth_model_path,
            depth_overlay_alpha=args.depth_overlay_alpha,
            depth_every_n_frames=max(1, int(args.depth_every_n_frames)),
            enable_drivable_seg=(not args.disable_drivable_seg),
            enable_relative_speed=(not args.disable_relative_speed),
            relative_speed_scale=max(1e-6, float(args.relative_speed_scale)),
        )
        pipeline = CLRNetOnnxPipeline(cfg)

        current_output = build_output_path(
            video_path=video_path,
            output_path=args.output,
            output_dir=args.output_dir,
            cut_height=cut_height,
            is_sweep=is_sweep,
        )

        print(f"\n=== Test cut_height={cut_height} ===")
        pipeline.run_video(
            video_path=video_path,
            output_path=current_output,
            show=(not args.no_show),
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
