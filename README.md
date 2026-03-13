# CUDA Batch Image Pipeline

This project is a simple but non-trivial CUDA image-processing assignment for the **CUDA at Scale Independent Project**. It processes a **batch of hundreds of grayscale images in one run** and compares a CPU pipeline against a CUDA GPU pipeline.

## What the project does

The program generates a synthetic dataset of grayscale images and applies this pipeline:

1. **Gaussian blur**
2. **Sobel edge detection**
3. **Binary thresholding**

The GPU implementation uses **custom CUDA kernels**. The blur and Sobel stages use **shared-memory tiling** to reduce repeated global-memory access for neighborhood operations.

## Why this fits the rubric

- Uses **GPU computation** through CUDA kernels
- Processes **hundreds of images** in one execution
- Includes a **README.md**
- Includes a **CLI** with arguments
- Includes support files for compiling and running: **Makefile** and **run.sh**
- Produces **proof artifacts**: output images, timing CSV, execution log, and terminal log

## Repository structure

```text
.
├── Makefile
├── README.md
├── run.sh
├── src/
│   └── main.cu
├── output/
└── proof/
```

## Build

```bash
make
```

## Run

```bash
./run.sh
```

The default run script does the following:

- builds the program
- captures `nvcc --version`
- captures `nvidia-smi`
- runs the program on **256 images of size 2048x2048**
- saves a terminal log to `proof/terminal_run.log`

## Command-line arguments

You can also run the executable directly:

```bash
./batch_image_pipeline --num-images 256 --width 2048 --height 2048 --threshold 110 --output-dir output
```

Available arguments:

- `--num-images N`
- `--width W`
- `--height H`
- `--threshold T`
- `--output-dir PATH`
- `--save-cpu`

## Output files

After running, the `output/` folder contains files such as:

- `input_0.pgm`
- `gpu_blur_0.pgm`
- `gpu_edges_0.pgm`
- `gpu_binary_0.pgm`
- `gpu_binary_1.pgm`
- `timings.csv`
- `execution_log.txt`

The `proof/` folder contains:

- `terminal_run.log`

## Suggested proof for peer review

To strengthen the submission, also add one screenshot showing:

- `nvcc --version`
- `nvidia-smi`
- the successful program run
- the produced output files

## Project description

This project demonstrates GPU-accelerated batch image processing with CUDA. A synthetic dataset of grayscale images is processed using a three-stage pipeline: Gaussian blur, Sobel edge detection, and binary thresholding. The same pipeline is also executed on the CPU so that execution times can be compared. The GPU version uses custom CUDA kernels, and shared memory is used in the stencil-based blur and Sobel stages to improve efficiency. The project shows how CUDA can accelerate image-processing workloads at scale while keeping the code straightforward and easy to review.
