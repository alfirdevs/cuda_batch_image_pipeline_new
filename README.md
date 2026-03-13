# CUDA Batch Image Pipeline

This project processes a batch of synthetic grayscale images using both CPU and GPU implementations.

## Pipeline
- Gaussian blur
- Sobel edge detection
- Binary thresholding

## GPU features
- Custom CUDA kernels
- Shared-memory tiling for blur and Sobel
- Batch processing of hundreds of images
- CPU vs GPU timing comparison

## Build
```bash
make
```

## Run
```bash
chmod +x run.sh
./run.sh
```

## Custom arguments
```bash
./batch_image_pipeline --num_images 256 --width 2048 --height 2048 --threshold 120
```

## Output
After running, the program writes:
- `output/*.pgm` sample images
- `execution_log.txt`
- `timings.csv`
- `proof/terminal_run.log`

## Notes
Do not upload the compiled executable to GitHub. Upload the source files and the generated proof artifacts.
