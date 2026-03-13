CXXFLAGS := -O3 -std=c++17
NVCC := nvcc
TARGET := batch_image_pipeline
SRC := src/main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf output/*

.PHONY: all clean
