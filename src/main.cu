#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kTileDim = 16;
constexpr int kBlurRadius = 1;
constexpr int kSobelRadius = 1;
constexpr int kDefaultNumImages = 256;
constexpr int kDefaultWidth = 2048;
constexpr int kDefaultHeight = 2048;
constexpr float kDefaultThreshold = 120.0f;

struct Options {
    int num_images = kDefaultNumImages;
    int width = kDefaultWidth;
    int height = kDefaultHeight;
    float threshold = kDefaultThreshold;
};

struct TimingResults {
    double cpu_total_ms = 0.0;
    double gpu_total_ms = 0.0;
    double gpu_h2d_ms = 0.0;
    double gpu_blur_ms = 0.0;
    double gpu_sobel_ms = 0.0;
    double gpu_threshold_ms = 0.0;
    double gpu_d2h_ms = 0.0;
};

inline int ClampInt(int value, int low, int high) {
    return (value < low) ? low : ((value > high) ? high : value);
}

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name
              << " [--num_images N] [--width W] [--height H] [--threshold T]\n";
}

Options ParseArguments(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--num_images" && i + 1 < argc) {
            options.num_images = std::stoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            options.width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            options.height = std::stoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            options.threshold = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (options.num_images <= 0 || options.width <= 0 || options.height <= 0) {
        std::cerr << "All dimensions and image count must be positive." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return options;
}

std::vector<uint8_t> GenerateSyntheticImages(int num_images, int width, int height) {
    const size_t pixels_per_image = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> images(static_cast<size_t>(num_images) * pixels_per_image);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> noise_dist(0, 30);

    for (int n = 0; n < num_images; ++n) {
        uint8_t* image = images.data() + static_cast<size_t>(n) * pixels_per_image;
        const int cx = width / 2 + (n % 13) * 7 - 40;
        const int cy = height / 2 + (n % 17) * 5 - 40;
        const int radius = 120 + (n % 5) * 20;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int dx = x - cx;
                const int dy = y - cy;
                const float dist = std::sqrt(static_cast<float>(dx * dx + dy * dy));
                int value = (x * 255) / width;

                if (dist < radius) {
                    value = 220;
                }
                if (((x / 64) + (y / 64) + n) % 2 == 0) {
                    value = std::min(255, value + 20);
                }
                value = std::min(255, std::max(0, value + noise_dist(rng)));
                image[static_cast<size_t>(y) * width + x] = static_cast<uint8_t>(value);
            }
        }
    }
    return images;
}

void SavePGM(const std::string& filename, const std::vector<uint8_t>& image,
             int width, int height) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.data()),
              static_cast<std::streamsize>(image.size()));
}

std::vector<uint8_t> FloatToUint8(const std::vector<float>& input) {
    std::vector<uint8_t> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        float value = input[i];
        value = std::max(0.0f, std::min(255.0f, value));
        output[i] = static_cast<uint8_t>(value);
    }
    return output;
}

std::vector<uint8_t> ThresholdToUint8(const std::vector<uint8_t>& input) {
    return input;
}

void CpuGaussianBlur(const uint8_t* input, float* output, int width, int height) {
    static constexpr float kernel[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f},
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            float weight_sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const int xx = ClampInt(x + kx, 0, width - 1);
                    const int yy = ClampInt(y + ky, 0, height - 1);
                    const float weight = kernel[ky + 1][kx + 1];
                    sum += static_cast<float>(input[yy * width + xx]) * weight;
                    weight_sum += weight;
                }
            }
            output[y * width + x] = sum / weight_sum;
        }
    }
}

void CpuSobel(const float* input, float* output, int width, int height) {
    static constexpr int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1},
    };
    static constexpr int gy[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1},
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sx = 0.0f;
            float sy = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const int xx = ClampInt(x + kx, 0, width - 1);
                    const int yy = ClampInt(y + ky, 0, height - 1);
                    const float pixel = input[yy * width + xx];
                    sx += pixel * static_cast<float>(gx[ky + 1][kx + 1]);
                    sy += pixel * static_cast<float>(gy[ky + 1][kx + 1]);
                }
            }
            output[y * width + x] = std::sqrt(sx * sx + sy * sy);
        }
    }
}

void CpuThreshold(const float* input, uint8_t* output, int width, int height, float threshold) {
    const size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t i = 0; i < num_pixels; ++i) {
        output[i] = (input[i] >= threshold) ? 255 : 0;
    }
}

__device__ __forceinline__ int DeviceClampInt(int value, int low, int high) {
    return (value < low) ? low : ((value > high) ? high : value);
}

__device__ inline void LoadTileUint8ToFloat(const uint8_t* input,
                                            float* tile_base,
                                            int tile_stride,
                                            int sx,
                                            int sy,
                                            int gx,
                                            int gy,
                                            int width,
                                            int height) {
    gx = DeviceClampInt(gx, 0, width - 1);
    gy = DeviceClampInt(gy, 0, height - 1);
    tile_base[sy * tile_stride + sx] = static_cast<float>(input[gy * width + gx]);
}

__device__ inline void LoadTileFloat(const float* input,
                                     float* tile_base,
                                     int tile_stride,
                                     int sx,
                                     int sy,
                                     int gx,
                                     int gy,
                                     int width,
                                     int height) {
    gx = DeviceClampInt(gx, 0, width - 1);
    gy = DeviceClampInt(gy, 0, height - 1);
    tile_base[sy * tile_stride + sx] = input[gy * width + gx];
}

__global__ void GaussianBlurKernel(const uint8_t* input, float* output, int width, int height) {
    __shared__ float tile[kTileDim + 2 * kBlurRadius][kTileDim + 2 * kBlurRadius];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int shared_x = threadIdx.x + kBlurRadius;
    const int shared_y = threadIdx.y + kBlurRadius;
    float* tile_base = &tile[0][0];
    const int tile_stride = kTileDim + 2 * kBlurRadius;

    LoadTileUint8ToFloat(input, tile_base, tile_stride, shared_x, shared_y, x, y, width, height);

    if (threadIdx.x < kBlurRadius) {
        LoadTileUint8ToFloat(input, tile_base, tile_stride, threadIdx.x, shared_y,
                             x - kBlurRadius, y, width, height);
        LoadTileUint8ToFloat(input, tile_base, tile_stride, shared_x + blockDim.x, shared_y,
                             x + blockDim.x, y, width, height);
    }
    if (threadIdx.y < kBlurRadius) {
        LoadTileUint8ToFloat(input, tile_base, tile_stride, shared_x, threadIdx.y,
                             x, y - kBlurRadius, width, height);
        LoadTileUint8ToFloat(input, tile_base, tile_stride, shared_x, shared_y + blockDim.y,
                             x, y + blockDim.y, width, height);
    }
    if (threadIdx.x < kBlurRadius && threadIdx.y < kBlurRadius) {
        LoadTileUint8ToFloat(input, tile_base, tile_stride, threadIdx.x, threadIdx.y,
                             x - kBlurRadius, y - kBlurRadius, width, height);
        LoadTileUint8ToFloat(input, tile_base, tile_stride, shared_x + blockDim.x, threadIdx.y,
                             x + blockDim.x, y - kBlurRadius, width, height);
        LoadTileUint8ToFloat(input, tile_base, tile_stride, threadIdx.x, shared_y + blockDim.y,
                             x - kBlurRadius, y + blockDim.y, width, height);
        LoadTileUint8ToFloat(input, tile_base, tile_stride,
                             shared_x + blockDim.x, shared_y + blockDim.y,
                             x + blockDim.x, y + blockDim.y, width, height);
    }

    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    static constexpr float kernel[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f},
    };

    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            const float weight = kernel[ky + 1][kx + 1];
            sum += tile[shared_y + ky][shared_x + kx] * weight;
            weight_sum += weight;
        }
    }
    output[y * width + x] = sum / weight_sum;
}

__global__ void SobelKernel(const float* input, float* output, int width, int height) {
    __shared__ float tile[kTileDim + 2 * kSobelRadius][kTileDim + 2 * kSobelRadius];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int shared_x = threadIdx.x + kSobelRadius;
    const int shared_y = threadIdx.y + kSobelRadius;
    float* tile_base = &tile[0][0];
    const int tile_stride = kTileDim + 2 * kSobelRadius;

    LoadTileFloat(input, tile_base, tile_stride, shared_x, shared_y, x, y, width, height);

    if (threadIdx.x < kSobelRadius) {
        LoadTileFloat(input, tile_base, tile_stride, threadIdx.x, shared_y,
                      x - kSobelRadius, y, width, height);
        LoadTileFloat(input, tile_base, tile_stride, shared_x + blockDim.x, shared_y,
                      x + blockDim.x, y, width, height);
    }
    if (threadIdx.y < kSobelRadius) {
        LoadTileFloat(input, tile_base, tile_stride, shared_x, threadIdx.y,
                      x, y - kSobelRadius, width, height);
        LoadTileFloat(input, tile_base, tile_stride, shared_x, shared_y + blockDim.y,
                      x, y + blockDim.y, width, height);
    }
    if (threadIdx.x < kSobelRadius && threadIdx.y < kSobelRadius) {
        LoadTileFloat(input, tile_base, tile_stride, threadIdx.x, threadIdx.y,
                      x - kSobelRadius, y - kSobelRadius, width, height);
        LoadTileFloat(input, tile_base, tile_stride, shared_x + blockDim.x, threadIdx.y,
                      x + blockDim.x, y - kSobelRadius, width, height);
        LoadTileFloat(input, tile_base, tile_stride, threadIdx.x, shared_y + blockDim.y,
                      x - kSobelRadius, y + blockDim.y, width, height);
        LoadTileFloat(input, tile_base, tile_stride,
                      shared_x + blockDim.x, shared_y + blockDim.y,
                      x + blockDim.x, y + blockDim.y, width, height);
    }

    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    static constexpr int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1},
    };
    static constexpr int gy[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1},
    };

    float sx = 0.0f;
    float sy = 0.0f;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            const float pixel = tile[shared_y + ky][shared_x + kx];
            sx += pixel * static_cast<float>(gx[ky + 1][kx + 1]);
            sy += pixel * static_cast<float>(gy[ky + 1][kx + 1]);
        }
    }

    output[y * width + x] = sqrtf(sx * sx + sy * sy);
}

__global__ void ThresholdKernel(const float* input, uint8_t* output,
                                int width, int height, float threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const float value = input[y * width + x];
        output[y * width + x] = (value >= threshold) ? 255 : 0;
    }
}

TimingResults RunCpuPipeline(const std::vector<uint8_t>& images,
                             std::vector<uint8_t>* sample_blur,
                             std::vector<uint8_t>* sample_sobel,
                             std::vector<uint8_t>* sample_thresh,
                             int num_images,
                             int width,
                             int height,
                             float threshold) {
    TimingResults timing;
    const size_t pixels_per_image = static_cast<size_t>(width) * static_cast<size_t>(height);

    std::vector<float> blur(pixels_per_image);
    std::vector<float> sobel(pixels_per_image);
    std::vector<uint8_t> binary(pixels_per_image);

    const auto start = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < num_images; ++n) {
        const uint8_t* image = images.data() + static_cast<size_t>(n) * pixels_per_image;
        CpuGaussianBlur(image, blur.data(), width, height);
        CpuSobel(blur.data(), sobel.data(), width, height);
        CpuThreshold(sobel.data(), binary.data(), width, height, threshold);

        if (n == 0) {
            *sample_blur = FloatToUint8(blur);
            *sample_sobel = FloatToUint8(sobel);
            *sample_thresh = ThresholdToUint8(binary);
        }
    }

    const auto end = std::chrono::high_resolution_clock::now();
    timing.cpu_total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return timing;
}

TimingResults RunGpuPipeline(const std::vector<uint8_t>& images,
                             std::vector<uint8_t>* sample_blur,
                             std::vector<uint8_t>* sample_sobel,
                             std::vector<uint8_t>* sample_thresh,
                             int num_images,
                             int width,
                             int height,
                             float threshold) {
    TimingResults timing;
    const size_t pixels_per_image = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t bytes_u8 = pixels_per_image * sizeof(uint8_t);
    const size_t bytes_f = pixels_per_image * sizeof(float);

    uint8_t* d_input = nullptr;
    float* d_blur = nullptr;
    float* d_sobel = nullptr;
    uint8_t* d_threshold = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes_u8));
    CUDA_CHECK(cudaMalloc(&d_blur, bytes_f));
    CUDA_CHECK(cudaMalloc(&d_sobel, bytes_f));
    CUDA_CHECK(cudaMalloc(&d_threshold, bytes_u8));

    dim3 block(kTileDim, kTileDim);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t total_start, total_stop;
    cudaEvent_t h2d_start, h2d_stop;
    cudaEvent_t blur_start, blur_stop;
    cudaEvent_t sobel_start, sobel_stop;
    cudaEvent_t thresh_start, thresh_stop;
    cudaEvent_t d2h_start, d2h_stop;

    CUDA_CHECK(cudaEventCreate(&total_start));
    CUDA_CHECK(cudaEventCreate(&total_stop));
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventCreate(&blur_start));
    CUDA_CHECK(cudaEventCreate(&blur_stop));
    CUDA_CHECK(cudaEventCreate(&sobel_start));
    CUDA_CHECK(cudaEventCreate(&sobel_stop));
    CUDA_CHECK(cudaEventCreate(&thresh_start));
    CUDA_CHECK(cudaEventCreate(&thresh_stop));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));

    std::vector<float> host_blur(pixels_per_image);
    std::vector<float> host_sobel(pixels_per_image);
    std::vector<uint8_t> host_thresh(pixels_per_image);

    CUDA_CHECK(cudaEventRecord(total_start));

    for (int n = 0; n < num_images; ++n) {
        const uint8_t* image = images.data() + static_cast<size_t>(n) * pixels_per_image;

        CUDA_CHECK(cudaEventRecord(h2d_start));
        CUDA_CHECK(cudaMemcpy(d_input, image, bytes_u8, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(h2d_stop));
        CUDA_CHECK(cudaEventSynchronize(h2d_stop));

        CUDA_CHECK(cudaEventRecord(blur_start));
        GaussianBlurKernel<<<grid, block>>>(d_input, d_blur, width, height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(blur_stop));
        CUDA_CHECK(cudaEventSynchronize(blur_stop));

        CUDA_CHECK(cudaEventRecord(sobel_start));
        SobelKernel<<<grid, block>>>(d_blur, d_sobel, width, height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(sobel_stop));
        CUDA_CHECK(cudaEventSynchronize(sobel_stop));

        CUDA_CHECK(cudaEventRecord(thresh_start));
        ThresholdKernel<<<grid, block>>>(d_sobel, d_threshold, width, height, threshold);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(thresh_stop));
        CUDA_CHECK(cudaEventSynchronize(thresh_stop));

        if (n == 0) {
            CUDA_CHECK(cudaEventRecord(d2h_start));
            CUDA_CHECK(cudaMemcpy(host_blur.data(), d_blur, bytes_f, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host_sobel.data(), d_sobel, bytes_f, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host_thresh.data(), d_threshold, bytes_u8, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(d2h_stop));
            CUDA_CHECK(cudaEventSynchronize(d2h_stop));

            *sample_blur = FloatToUint8(host_blur);
            *sample_sobel = FloatToUint8(host_sobel);
            *sample_thresh = ThresholdToUint8(host_thresh);
        } else {
            float temp_ms = 0.0f;
            CUDA_CHECK(cudaEventRecord(d2h_start));
            CUDA_CHECK(cudaEventRecord(d2h_stop));
            CUDA_CHECK(cudaEventSynchronize(d2h_stop));
            CUDA_CHECK(cudaEventElapsedTime(&temp_ms, d2h_start, d2h_stop));
            timing.gpu_d2h_ms += temp_ms;
        }

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, h2d_start, h2d_stop));
        timing.gpu_h2d_ms += ms;

        CUDA_CHECK(cudaEventElapsedTime(&ms, blur_start, blur_stop));
        timing.gpu_blur_ms += ms;

        CUDA_CHECK(cudaEventElapsedTime(&ms, sobel_start, sobel_stop));
        timing.gpu_sobel_ms += ms;

        CUDA_CHECK(cudaEventElapsedTime(&ms, thresh_start, thresh_stop));
        timing.gpu_threshold_ms += ms;

        if (n == 0) {
            CUDA_CHECK(cudaEventElapsedTime(&ms, d2h_start, d2h_stop));
            timing.gpu_d2h_ms += ms;
        }
    }

    CUDA_CHECK(cudaEventRecord(total_stop));
    CUDA_CHECK(cudaEventSynchronize(total_stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_start, total_stop));
    timing.gpu_total_ms = total_ms;

    CUDA_CHECK(cudaEventDestroy(total_start));
    CUDA_CHECK(cudaEventDestroy(total_stop));
    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(blur_start));
    CUDA_CHECK(cudaEventDestroy(blur_stop));
    CUDA_CHECK(cudaEventDestroy(sobel_start));
    CUDA_CHECK(cudaEventDestroy(sobel_stop));
    CUDA_CHECK(cudaEventDestroy(thresh_start));
    CUDA_CHECK(cudaEventDestroy(thresh_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_blur));
    CUDA_CHECK(cudaFree(d_sobel));
    CUDA_CHECK(cudaFree(d_threshold));

    return timing;
}

std::string GetGpuName() {
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return std::string(prop.name);
}

void WriteExecutionLog(const std::string& filename,
                       const Options& options,
                       const TimingResults& timing,
                       const std::string& gpu_name) {
    std::ofstream out(filename);
    out << "CUDA Batch Image Pipeline Execution Log\n";
    out << "======================================\n";
    out << "GPU: " << gpu_name << "\n";
    out << "Number of images: " << options.num_images << "\n";
    out << "Image size: " << options.width << " x " << options.height << "\n";
    out << "Threshold: " << options.threshold << "\n\n";

    out << std::fixed << std::setprecision(3);
    out << "CPU total time (ms): " << timing.cpu_total_ms << "\n";
    out << "GPU total time (ms): " << timing.gpu_total_ms << "\n";
    out << "GPU H2D total (ms): " << timing.gpu_h2d_ms << "\n";
    out << "GPU blur total (ms): " << timing.gpu_blur_ms << "\n";
    out << "GPU sobel total (ms): " << timing.gpu_sobel_ms << "\n";
    out << "GPU threshold total (ms): " << timing.gpu_threshold_ms << "\n";
    out << "GPU D2H total (ms): " << timing.gpu_d2h_ms << "\n";

    if (timing.gpu_total_ms > 0.0) {
        out << "Speedup (CPU/GPU): " << (timing.cpu_total_ms / timing.gpu_total_ms) << "x\n";
    }
}

void WriteTimingCsv(const std::string& filename,
                    const Options& options,
                    const TimingResults& timing) {
    std::ofstream out(filename);
    out << "num_images,width,height,threshold,cpu_total_ms,gpu_total_ms,"
           "gpu_h2d_ms,gpu_blur_ms,gpu_sobel_ms,gpu_threshold_ms,gpu_d2h_ms,speedup\n";
    const double speedup = (timing.gpu_total_ms > 0.0)
                               ? (timing.cpu_total_ms / timing.gpu_total_ms)
                               : 0.0;
    out << options.num_images << ','
        << options.width << ','
        << options.height << ','
        << options.threshold << ','
        << timing.cpu_total_ms << ','
        << timing.gpu_total_ms << ','
        << timing.gpu_h2d_ms << ','
        << timing.gpu_blur_ms << ','
        << timing.gpu_sobel_ms << ','
        << timing.gpu_threshold_ms << ','
        << timing.gpu_d2h_ms << ','
        << speedup << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = ParseArguments(argc, argv);

    fs::create_directories("output");
    fs::create_directories("proof");

    std::cout << "Generating " << options.num_images << " synthetic images of size "
              << options.width << "x" << options.height << "..." << std::endl;

    const std::vector<uint8_t> images =
        GenerateSyntheticImages(options.num_images, options.width, options.height);

    std::vector<uint8_t> cpu_blur_sample;
    std::vector<uint8_t> cpu_sobel_sample;
    std::vector<uint8_t> cpu_thresh_sample;

    std::vector<uint8_t> gpu_blur_sample;
    std::vector<uint8_t> gpu_sobel_sample;
    std::vector<uint8_t> gpu_thresh_sample;

    std::cout << "Running CPU pipeline..." << std::endl;
    TimingResults cpu_timing = RunCpuPipeline(images,
                                              &cpu_blur_sample,
                                              &cpu_sobel_sample,
                                              &cpu_thresh_sample,
                                              options.num_images,
                                              options.width,
                                              options.height,
                                              options.threshold);

    std::cout << "Running GPU pipeline..." << std::endl;
    TimingResults gpu_timing = RunGpuPipeline(images,
                                              &gpu_blur_sample,
                                              &gpu_sobel_sample,
                                              &gpu_thresh_sample,
                                              options.num_images,
                                              options.width,
                                              options.height,
                                              options.threshold);

    TimingResults total_timing;
    total_timing.cpu_total_ms = cpu_timing.cpu_total_ms;
    total_timing.gpu_total_ms = gpu_timing.gpu_total_ms;
    total_timing.gpu_h2d_ms = gpu_timing.gpu_h2d_ms;
    total_timing.gpu_blur_ms = gpu_timing.gpu_blur_ms;
    total_timing.gpu_sobel_ms = gpu_timing.gpu_sobel_ms;
    total_timing.gpu_threshold_ms = gpu_timing.gpu_threshold_ms;
    total_timing.gpu_d2h_ms = gpu_timing.gpu_d2h_ms;

    SavePGM("output/input_sample.pgm",
            std::vector<uint8_t>(images.begin(),
                                 images.begin() + static_cast<long long>(options.width) * options.height),
            options.width, options.height);

    SavePGM("output/cpu_blur_sample.pgm", cpu_blur_sample, options.width, options.height);
    SavePGM("output/cpu_sobel_sample.pgm", cpu_sobel_sample, options.width, options.height);
    SavePGM("output/cpu_threshold_sample.pgm", cpu_thresh_sample, options.width, options.height);

    SavePGM("output/gpu_blur_sample.pgm", gpu_blur_sample, options.width, options.height);
    SavePGM("output/gpu_sobel_sample.pgm", gpu_sobel_sample, options.width, options.height);
    SavePGM("output/gpu_threshold_sample.pgm", gpu_thresh_sample, options.width, options.height);

    const std::string gpu_name = GetGpuName();
    WriteExecutionLog("execution_log.txt", options, total_timing, gpu_name);
    WriteTimingCsv("timings.csv", options, total_timing);

    std::ofstream terminal_log("proof/terminal_run.log");
    terminal_log << "CUDA Batch Image Pipeline\n";
    terminal_log << "GPU: " << gpu_name << '\n';
    terminal_log << "Images: " << options.num_images << '\n';
    terminal_log << "Dimensions: " << options.width << " x " << options.height << '\n';
    terminal_log << std::fixed << std::setprecision(3);
    terminal_log << "CPU total time (ms): " << total_timing.cpu_total_ms << '\n';
    terminal_log << "GPU total time (ms): " << total_timing.gpu_total_ms << '\n';
    if (total_timing.gpu_total_ms > 0.0) {
        terminal_log << "Speedup (CPU/GPU): "
                     << (total_timing.cpu_total_ms / total_timing.gpu_total_ms) << "x\n";
    }

    std::cout << "\nDone.\n";
    std::cout << "GPU: " << gpu_name << "\n";
    std::cout << "CPU total time (ms): " << std::fixed << std::setprecision(3)
              << total_timing.cpu_total_ms << "\n";
    std::cout << "GPU total time (ms): " << total_timing.gpu_total_ms << "\n";
    if (total_timing.gpu_total_ms > 0.0) {
        std::cout << "Speedup (CPU/GPU): "
                  << (total_timing.cpu_total_ms / total_timing.gpu_total_ms) << "x\n";
    }
    std::cout << "Saved outputs to output/\n";
    std::cout << "Saved logs to execution_log.txt, timings.csv, and proof/terminal_run.log\n";

    return 0;
}
