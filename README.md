# Depth-Anything-V2-Small: PyTorch to Embedded C++ Deployment

Comparing **Manual C++** vs **MATLAB Coder** for deploying a 24.7M-parameter Vision Transformer model to standalone C++ code.

## Model

- **Depth-Anything-V2-Small**: Monocular depth estimation
- **Architecture**: DINOv2-Small ViT encoder (12 blocks, dim=384) + DPT decoder (4-scale RefineNet)
- **Input**: `[1, 3, 518, 784]` float32 (RGB image)
- **Output**: `[1, 518, 784]` float32 (depth map)
- **Parameters**: 24,710,849 (94.3 MB)

## Results

| Metric | Manual C++ | MATLAB Coder |
|--------|-----------|-------------|
| Inference Time | **13,564 ms** | 16,073 ms |
| Relative RMSE | 2.79e-1 | **5.09e-7** |
| Max Absolute Error | 1.294 | **4.53e-6** |
| C++ Lines | ~600 | 44,361 |
| Development Time | Hours | **Minutes** |

**MATLAB Coder is 548,000x more accurate** with fully automated code generation.

## Project Structure

```
depth_deploy/
  manual_cpp/          # Approach A: hand-written C++ implementation
    depth_anything_v2.h  # Model implementation (~600 lines)
    main.cpp             # Test harness and benchmark
  matlab_coder/        # Approach B: MATLAB Coder generated code
    depth_infer.m        # 7-line entry point for codegen
    run_codegen.m        # Code generation script
    run_benchmark.m      # Benchmark script
    codegen_out/         # Generated C++ (44K lines + 73 weight files)
  reference/           # PyTorch reference I/O for validation
    architecture.json    # Model architecture metadata
    io_meta.json         # Input/output specifications
    input.bin            # Reference input (seed=42)
    output.bin           # Reference output
  weights/             # Extracted model weights (234 tensors)
  webapp/              # Interactive comparison app (MathWorks theme)
    index.html           # Self-contained single-page app
  report/              # PDF report
    Depth_Anything_V2_Deployment_Report.pdf
```

## Quick Start

### Manual C++ (Approach A)
```bash
cd depth_deploy/manual_cpp
clang++ -std=c++17 -O3 -framework Accelerate -o depth_test main.cpp
./depth_test ../weights ../reference
```

### MATLAB Coder (Approach B)
```matlab
cd depth_deploy/matlab_coder
run_codegen     % Generates C++ from .pt2
run_benchmark   % Builds MEX and benchmarks
```

### Interactive App
```bash
python3 -m http.server 8080 --directory depth_deploy/webapp
# Open http://localhost:8080
```

## Requirements

- macOS with Apple Silicon (for Accelerate framework)
- MATLAB R2026a with:
  - MATLAB Coder
  - Embedded Coder
  - Deep Learning Toolbox
  - Coder Support Package for PyTorch
- Depth-Anything-V2-Small .pt2 file (not included, ~100 MB)

## License

This project is for engineering evaluation and testing purposes.
