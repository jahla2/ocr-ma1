from __future__ import annotations

import onnxruntime as ort


def main() -> None:
    providers = ort.get_available_providers()
    print("ONNX Runtime providers:")
    for provider in providers:
        print(f" - {provider}")

    if "CUDAExecutionProvider" in providers:
        print("\nCUDAExecutionProvider is available: GPU mode is ready.")
    else:
        print(
            "\nCUDAExecutionProvider is NOT available. "
            "Install onnxruntime-gpu and verify your NVIDIA/CUDA runtime."
        )


if __name__ == "__main__":
    main()
