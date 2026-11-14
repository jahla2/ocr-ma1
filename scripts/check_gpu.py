from __future__ import annotations

import onnxruntime as ort


def main() -> None:
    print(f"ONNX Runtime version: {ort.__version__}")

    if hasattr(ort, "preload_dlls"):
        print("\nPreloading CUDA/cuDNN DLLs from Python site-packages...")
        ort.preload_dlls(directory="")

    providers = ort.get_available_providers()
    print("\nONNX Runtime providers:")
    for provider in providers:
        print(f" - {provider}")

    if hasattr(ort, "print_debug_info"):
        print("\nRuntime DLL diagnostics:")
        ort.print_debug_info()

    if "CUDAExecutionProvider" in providers:
        print("\nCUDAExecutionProvider is available.")
        print("If cudnn64_9.dll is shown as loaded above, GPU inference is ready.")
    else:
        print(
            "\nCUDAExecutionProvider is NOT available. "
            "Install the GPU extra and verify your NVIDIA driver."
        )


if __name__ == "__main__":
    main()
