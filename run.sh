#!/usr/bin/env bash
set -e

make
./batch_image_pipeline --num_images 256 --width 2048 --height 2048 --threshold 120
