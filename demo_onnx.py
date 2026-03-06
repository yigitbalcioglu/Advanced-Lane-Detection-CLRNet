import argparse

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
    parser.add_argument("--disable-bev", action="store_true", help="BEV donusumunu kapat")
    parser.add_argument("--speed-mps", type=float, default=12.0, help="Stanley icin hiz (m/s)")
    parser.add_argument("--lane-width-m", type=float, default=3.5, help="Standart serit genisligi")
    parser.add_argument("--no-dashboard", action="store_true", help="Sag panel dashboardu kapat")
    parser.add_argument("--cut-height", type=int, default=400, help="Ustten kirpma miktari")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = resolve_video_path(args.video_name, args.videos_dir)

    cfg = OnnxPipelineConfig(
        model_path=args.onnx,
        cut_height=args.cut_height,
        conf_threshold=args.conf_threshold,
        force_gpu=(not args.allow_cpu),
        use_bev=(not args.disable_bev),
        speed_mps=args.speed_mps,
        lane_width_m=args.lane_width_m,
        show_dashboard=(not args.no_dashboard),
    )
    pipeline = CLRNetOnnxPipeline(cfg)

    pipeline.run_video(
        video_path=video_path,
        output_path=args.output,
        show=(not args.no_show),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
