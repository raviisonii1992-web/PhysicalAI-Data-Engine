import platform
import shutil
import subprocess
import sys

import cv2
import torch

def check_python():
    print("Python")
    print(f"  Version : {sys.version.split()[0]}")
    print(f"  Path    : {sys.executable}")

def check_system():
    print("\nSystem")
    print(f"  OS      : {platform.system()}")
    print(f"  Release : {platform.release()}")
    print(f"  Machine : {platform.machine()}")

def check_pytorch():
    print("\nPyTorch")
    print(f"  Version        : {torch.__version__}")
    print(f"  CUDA Available : {torch.cuda.is_available()}")
    print(f"  CUDA Runtime   : {torch.version.cuda}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / (1024 ** 3)

        print(f"  GPU            : {gpu_name}")
        print(f"  GPU Memory     : {total_memory_gb:.2f} GB")

def check_opencv():
    print("\nOpenCV")
    print(f"  Version : {cv2.__version__}")

    build_info = cv2.getBuildInformation()

    if "FFMPEG:                      YES" in build_info:
        print("  FFmpeg Support : YES")
    else:
        print("  FFmpeg Support : CHECK REQUIRED")


def check_ffmpeg():
    print("\nFFmpeg")

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        print("  Status : NOT FOUND")
        return

    print(f"  Path   : {ffmpeg_path}")

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )

        first_line = result.stdout.splitlines()[0]
        print(f"  Version: {first_line}")

    except subprocess.CalledProcessError:
        print("  Status : ERROR")


def gpu_test():
    print("\nGPU Computation Test")

    if not torch.cuda.is_available():
        print("  Skipped: CUDA is unavailable")
        return

    x = torch.rand((1000, 1000), device="cuda")
    y = torch.matmul(x, x)

    print(f"  Tensor Device : {y.device}")
    print("  Result        : SUCCESS")


def main():
    print("=" * 55)
    print("PhysicalAI Data Engine - Environment Check")
    print("=" * 55)

    check_python()
    check_system()
    check_pytorch()
    check_opencv()
    check_ffmpeg()
    gpu_test()

    print("\n" + "=" * 55)
    print("Environment check completed")
    print("=" * 55)


if __name__ == "__main__":
    main()