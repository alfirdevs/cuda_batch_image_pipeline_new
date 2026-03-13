TARGET = batch_image_pipeline
SRC = src/main.cu

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f execution_log.txt timings.csv
	rm -rf output proof

run: $(TARGET)
	./$(TARGET) --num_images 256 --width 2048 --height 2048 --threshold 120
