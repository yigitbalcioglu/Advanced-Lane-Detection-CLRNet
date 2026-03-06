import argparse
import os
import shutil
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX -> TensorRT engine export helper")
    parser.add_argument("--onnx", required=True, help="Input ONNX path")
    parser.add_argument("--engine", required=True, help="Output TensorRT engine path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 engine")
    parser.add_argument("--workspace", type=int, default=2048, help="Workspace size in MB")
    parser.add_argument("--trtexec", default="trtexec", help="Path to trtexec executable")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX bulunamadi: {args.onnx}")

    trtexec_path = shutil.which(args.trtexec) if args.trtexec == "trtexec" else args.trtexec
    if not trtexec_path or not os.path.exists(trtexec_path):
        raise FileNotFoundError(
            "trtexec bulunamadi. TensorRT kurup PATH'e ekleyin veya --trtexec ile tam yol verin."
        )

    cmd = [
        trtexec_path,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        f"--workspace={args.workspace}",
    ]
    if args.fp16:
        cmd.append("--fp16")

    print("Komut:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"TensorRT engine kaydedildi: {args.engine}")


if __name__ == "__main__":
    main()
