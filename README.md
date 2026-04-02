# Depth-Anything-V2-Small: PyTorch to Embedded C++ Deployment

Comparing **Manual C++** vs **MATLAB Coder** for deploying a 24.7M-parameter Vision Transformer model to standalone C++ code, with analysis of real-world embedded deployment targets.

**[Interactive App](https://aturevsk.github.io/depth-anything-v2-cpp-deploy/)** | **[PDF Report](depth_deploy/report/Depth_Anything_V2_Deployment_Report.pdf)**

## Model

- **Depth-Anything-V2-Small**: Monocular depth estimation (DINOv2-Small + DPT)
- **Input**: `[1, 3, 518, 784]` float32 (RGB image)
- **Output**: `[1, 518, 784]` float32 (depth map)
- **Parameters**: 24,710,849 (94.3 MB)

## Three Implementations

| Metric | Manual C++ (BLAS) | S32-Compatible | MATLAB Coder |
|--------|:-----------------:|:--------------:|:------------:|
| Inference Time | **1,043 ms** | 26,411 ms | 12,918 ms |
| Relative RMSE | 2.24e-3 | 2.24e-3 | **5.57e-7** |
| External BLAS | Apple Accelerate | **None** | **None** |
| Static Allocation | No (heap) | **Yes** | **Yes** |
| STL Containers | vector, string | **None** | **None** |
| MISRA-Friendly | No | **Yes** | **Yes** |
| Code Traceability | None | [pt2:] comments | **Automatic** |
| C++ Lines | ~693 | ~940 | 44,361 |

### Key Insight: BLAS Determines Portability

The Manual C++ speed advantage (12.3x over MATLAB Coder) exists **only because Apple Accelerate BLAS is available**. On safety-certified ECUs (AUTOSAR/ISO 26262) where MATLAB Coder is designed to target, BLAS does not exist. The S32-compatible version proves this: without BLAS, it runs at 26s — slower than MATLAB Coder's 13s, and without certification.

## Project Structure

```
depth_deploy/
  manual_cpp/              # Manual C++ implementations
    depth_anything_v2.h      # Original (BLAS-optimized, ~693 lines)
    main.cpp                 # Test harness (Accelerate framework)
    depth_anything_v2_s32.h  # S32-compatible (no BLAS, static alloc, ~940 lines)
    main_s32.cpp             # S32 test harness (no external dependencies)
  matlab_coder/            # MATLAB Coder generated code
    depth_infer.m            # 7-line entry point
    run_codegen.m            # Code generation script
    codegen_out/             # Generated C++ (44K lines)
  reference/               # PyTorch reference I/O
  weights/                 # Model weights (234 tensors, 94 MB)
  webapp/                  # Interactive comparison app
    index.html               # Self-contained single-page app (9 tabs)
  report/                  # PDF report
    generate_report.py       # Report generator (ReportLab)
    Depth_Anything_V2_Deployment_Report.pdf
docs/                      # GitHub Pages (copy of webapp)
```

## Quick Start

### Manual C++ — BLAS-Optimized (macOS)
```bash
cd depth_deploy/manual_cpp
clang++ -std=c++17 -O3 -DACCELERATE_NEW_LAPACK -framework Accelerate -o depth_test main.cpp
./depth_test ../weights ../reference
# → 1,043 ms, RMSE 2.24e-3
```

### Manual C++ — S32-Compatible (any platform)
```bash
cd depth_deploy/manual_cpp
clang++ -std=c++14 -O3 -ffp-contract=off -o depth_test_s32 main_s32.cpp
./depth_test_s32 ../weights ../reference
# → 26,411 ms, RMSE 2.24e-3, zero external dependencies
```

### MATLAB Coder
```matlab
cd depth_deploy/matlab_coder
run_codegen     % Generates C++ from .pt2
run_benchmark   % Builds MEX and benchmarks
% → 12,918 ms, RMSE 5.57e-7
```

### Interactive App
Open `depth_deploy/webapp/index.html` directly in a browser, or visit the [GitHub Pages site](https://aturevsk.github.io/depth-anything-v2-cpp-deploy/).

## Embedded Deployment Targets

| Target Class | Correct Approach | Why |
|-------------|-----------------|-----|
| Jetson / RK3588 / mobile SoC | ONNX + TensorRT/SNPE/Core ML | NPU gives 5-100ms INT8 |
| Safety-certified ECU (AUTOSAR) | MATLAB Coder + Embedded Coder | MISRA, traceability built-in |
| Linux ARM board (prototyping) | Manual C++ + BLAS | 1,043ms FP32 for development |
| NXP S32G / Renesas R-Car | S32-compatible C++ or MATLAB Coder | No BLAS available, ~13-26s FP32 |

**Note on Qualcomm SA8540P**: This is a high-performance SoC with GPU/Adreno and Hexagon NPU — use ONNX+SNPE, not MATLAB Coder. Real MATLAB Coder targets are bare Cortex-A domain controllers (Renesas R-Car, NXP S32, TI TDA4VM).

## Requirements

- **S32-compatible build**: Any C++14 compiler (clang, gcc, armclang) — no platform dependencies
- **BLAS-optimized build**: macOS with Apple Silicon (Accelerate framework)
- **MATLAB Coder**: R2026a with Deep Learning Toolbox, MATLAB Coder, Embedded Coder, Coder Support Package for PyTorch

## License

This project is for engineering evaluation and testing purposes.
