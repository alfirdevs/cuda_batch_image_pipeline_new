#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " -> " << cudaGetErrorString(err__) << std::endl;  \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

namespace {

constexpr int kTileDim = 16;
constexpr int kBlurRadius = 1;
constexpr int kSobelRadius = 1;

struct Config {
    int num_images = 256;
    int width = 2048;
    int height = 2048;
    int threshold = 100;
    bool save_cpu_outputs = false;
    std::string output_dir = "output";
};

struct Timings {
    double cpu_blur_ms = 0.0;
    double cpu_sobel_ms = 0.0;
    double cpu_threshold_ms = 0.0;
    double cpu_total_ms = 0.0;
    double gpu_blur_ms = 0.0;
    double gpu_sobel_ms = 0.0;
    double gpu_threshold_ms = 0.0;
    double gpu_total_ms = 0.0;
};

__host__ __device__ int ClampInt(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --num-images N      Number of synthetic images (default: 256)\n"
              << "  --width W           Image width (default: 2048)\n"
              << "  --height H          Image height (default: 2048)\n"
              << "  --threshold T       Edge threshold 0-255 (default: 100)\n"
              << "  --output-dir PATH   Output directory (default: output)\n"
              << "  --save-cpu          Save CPU sample outputs too\n"
              << "  --help              Show this message\n";
}

Config ParseArgs(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--num-images") {
            cfg.num_images = std::stoi(require_value(arg));
        } else if (arg == "--width") {
            cfg.width = std::stoi(require_value(arg));
        } else if (arg == "--height") {
            cfg.height = std::stoi(require_value(arg));
        } else if (arg == "--threshold") {
            cfg.threshold = std::stoi(require_value(arg));
        } else if (arg == "--output-dir") {
            cfg.output_dir = require_value(arg);
        } else if (arg == "--save-cpu") {
            cfg.save_cpu_outputs = true;
        } else if (arg == "--help") {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.num_images <= 0 || cfg.width <= 0 || cfg.height <= 0) {
        throw std::runtime_error("Image counts and dimensions must be positive.");
    }
    cfg.threshold = ClampInt(cfg.threshold, 0, 255);
    return cfg;
}

std::vector<uint8_t> GenerateSyntheticBatch(const Config& cfg) {
    const size_t pixels_per_image = static_cast<size_t>(cfg.width) * cfg.height;
    std::vector<uint8_t> batch(static_cast<size_t>(cfg.num_images) * pixels_per_image);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> noise_dist(0, 24);

    for (int img = 0; img < cfg.num_images; ++img) {
        uint8_t* image = batch.data() + static_cast<size_t>(img) * pixels_per_image;
        const int center_x = cfg.width / 2 + (img % 11 - 5) * (cfg.width / 20);
        const int center_y = cfg.height / 2 + (img % 13 - 6) * (cfg.height / 24);
        const int radius = std::max(24, std::min(cfg.width, cfg.height) / 6 + (img % 5) * 10);

        for (int y = 0; y < cfg.height; ++y) {
            for (int x = 0; x < cfg.width; ++x) {
                const int idx = y * cfg.width + x;
                const float gradient = 255.0f * static_cast<float>(x) / std::max(1, cfg.width - 1);
                const int dx = x - center_x;
                const int dy = y - center_y;
                const float dist = std::sqrt(static_cast<float>(dx * dx + dy * dy));
                const float circle = dist < radius ? 120.0f : 0.0f;
                const float stripe = ((x / 32 + y / 32 + img) % 2 == 0) ? 35.0f : 0.0f;
                const int noise = noise_dist(rng);
                const int value = static_cast<int>(gradient * 0.45f + circle + stripe + noise);
                image[idx] = static_cast<uint8_t>(ClampInt(value, 0, 255));
            }
        }
    }
    return batch;
}

void SavePgm(const std::string& path, const uint8_t* data, int width, int height) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    out << "P5\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(width) * height);
}

void WriteTimingCsv(const std::string& path, const Timings& t) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write timings CSV: " + path);
    }
    out << "stage,cpu_ms,gpu_ms,speedup\n";
    auto write_row = [&](const std::string& stage, double cpu_ms, double gpu_ms) {
        const double speedup = gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
        out << stage << ',' << std::fixed << std::setprecision(3)
            << cpu_ms << ',' << gpu_ms << ',' << speedup << '\n';
    };
    write_row("blur", t.cpu_blur_ms, t.gpu_blur_ms);
    write_row("sobel", t.cpu_sobel_ms, t.gpu_sobel_ms);
    write_row("threshold", t.cpu_threshold_ms, t.gpu_threshold_ms);
    write_row("total", t.cpu_total_ms, t.gpu_total_ms);
}

void WriteExecutionLog(const std::string& path,
                       const Config& cfg,
                       const Timings& t,
                       const cudaDeviceProp& prop) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write execution log: " + path);
    }

    const size_t total_pixels = static_cast<size_t>(cfg.num_images) * cfg.width * cfg.height;
    const double gpu_speedup = t.gpu_total_ms > 0.0 ? t.cpu_total_ms / t.gpu_total_ms : 0.0;

    out << "CUDA Batch Image Pipeline Execution Log\n";
    out << "=====================================\n";
    out << "Images processed: " << cfg.num_images << '\n';
    out << "Image size: " << cfg.width << 'x' << cfg.height << '\n';
    out << "Total pixels: " << total_pixels << '\n';
    out << "Threshold: " << cfg.threshold << '\n';
    out << "Output directory: " << cfg.output_dir << "\n\n";

    out << "GPU device: " << prop.name << '\n';
    out << "Compute capability: " << prop.major << '.' << prop.minor << '\n';
    out << "Global memory (MB): " << (prop.totalGlobalMem / (1024 * 1024)) << '\n';
    out << "Shared memory per block (KB): " << (prop.sharedMemPerBlock / 1024) << '\n';
    out << "Multiprocessor count: " << prop.multiProcessorCount << "\n\n";

    out << std::fixed << std::setprecision(3);
    out << "CPU blur ms: " << t.cpu_blur_ms << '\n';
    out << "CPU sobel ms: " << t.cpu_sobel_ms << '\n';
    out << "CPU threshold ms: " << t.cpu_threshold_ms << '\n';
    out << "CPU total ms: " << t.cpu_total_ms << "\n\n";

    out << "GPU blur ms: " << t.gpu_blur_ms << '\n';
    out << "GPU sobel ms: " << t.gpu_sobel_ms << '\n';
    out << "GPU threshold ms: " << t.gpu_threshold_ms << '\n';
    out << "GPU total ms: " << t.gpu_total_ms << "\n\n";

    out << "Overall speedup (CPU/GPU): " << gpu_speedup << 'x' << '\n';
}

void CpuBlur(const uint8_t* input, float* output, int width, int height) {
    static constexpr int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1},
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            float weight_sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const int xx = ClampInt(x + kx, 0, width - 1);
                    const int yy = ClampInt(y + ky, 0, height - 1);
                    const int weight = kernel[ky + 1][kx + 1];
                    sum += static_cast<float>(input[yy * width + xx]) * weight;
                    weight_sum += static_cast<float>(weight);
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
                    sx += pixel * gx[ky + 1][kx + 1];
                    sy += pixel * gy[ky + 1][kx + 1];
                }
            }
            output[y * width + x] = std::sqrt(sx * sx + sy * sy);
        }
    }
}

void CpuThreshold(const float* input, uint8_t* output, int width, int height, int threshold) {
    const int count = width * height;
    for (int i = 0; i < count; ++i) {
        output[i] = input[i] >= threshold ? 255 : 0;
    }
}

__global__ void GaussianBlurKernel(const uint8_t* input, float* output, int width, int height) {
    __shared__ float tile[kTileDim + 2 * kBlurRadius][kTileDim + 2 * kBlurRadius];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int shared_x = threadIdx.x + kBlurRadius;
    const int shared_y = threadIdx.y + kBlurRadius;

    auto load_to_shared = [&](int sx, int sy, int gx, int gy) {
        gx = ClampInt(gx, 0, width - 1);
        gy = ClampInt(gy, 0, height - 1);
        tile[sy][sx] = static_cast<float>(input[gy * width + gx]);
    };

    load_to_shared(shared_x, shared_y, x, y);

    if (threadIdx.x < kBlurRadius) {
        load_to_shared(threadIdx.x, shared_y, x - kBlurRadius, y);
        load_to_shared(shared_x + blockDim.x, shared_y, x + blockDim.x, y);
    }
    if (threadIdx.y < kBlurRadius) {
        load_to_shared(shared_x, threadIdx.y, x, y - kBlurRadius);
        load_to_shared(shared_x, shared_y + blockDim.y, x, y + blockDim.y);
    }
    if (threadIdx.x < kBlurRadius && threadIdx.y < kBlurRadius) {
        load_to_shared(threadIdx.x, threadIdx.y, x - kBlurRadius, y - kBlurRadius);
        load_to_shared(shared_x + blockDim.x, threadIdx.y, x + blockDim.x, y - kBlurRadius);
        load_to_shared(threadIdx.x, shared_y + blockDim.y, x - kBlurRadius, y + blockDim.y);
        load_to_shared(shared_x + blockDim.x,
                       shared_y + blockDim.y,
                       x + blockDim.x,
                       y + blockDim.y);
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

    auto load_to_shared = [&](int sx, int sy, int gx, int gy) {
        gx = ClampInt(gx, 0, width - 1);
        gy = ClampInt(gy, 0, height - 1);
        tile[sy][sx] = input[gy * width + gx];
    };

    load_to_shared(shared_x, shared_y, x, y);

    if (threadIdx.x < kSobelRadius) {
        load_to_shared(threadIdx.x, shared_y, x - kSobelRadius, y);
        load_to_shared(shared_x + blockDim.x, shared_y, x + blockDim.x, y);
    }
    if (threadIdx.y < kSobelRadius) {
        load_to_shared(shared_x, threadIdx.y, x, y - kSobelRadius);
        load_to_shared(shared_x, shared_y + blockDim.y, x, y + blockDim.y);
    }
    if (threadIdx.x < kSobelRadius && threadIdx.y < kSobelRadius) {
        load_to_shared(threadIdx.x, threadIdx.y, x - kSobelRadius, y - kSobelRadius);
        load_to_shared(shared_x + blockDim.x, threadIdx.y, x + blockDim.x, y - kSobelRadius);
        load_to_shared(threadIdx.x, shared_y + blockDim.y, x - kSobelRadius, y + blockDim.y);
        load_to_shared(shared_x + blockDim.x,
                       shared_y + blockDim.y,
                       x + blockDim.x,
                       y + blockDim.y);
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
            sx += pixel * gx[ky + 1][kx + 1];
            sy += pixel * gy[ky + 1][kx + 1];
        }
    }

    output[y * width + x] = sqrtf(sx * sx + sy * sy);
}

__global__ void ThresholdKernel(const float* input, uint8_t* output, int count, int threshold) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = input[idx] >= static_cast<float>(threshold) ? 255 : 0;
    }
}

Timings RunCpuPipeline(const Config& cfg,
                       const std::vector<uint8_t>& input,
                       std::vector<float>& blurred,
                       std::vector<float>& edges,
                       std::vector<uint8_t>& binary) {
    Timings t;
    const size_t pixels_per_image = static_cast<size_t>(cfg.width) * cfg.height;

    auto cpu_start_total = std::chrono::high_resolution_clock::now();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.num_images; ++i) {
        CpuBlur(input.data() + i * pixels_per_image,
                blurred.data() + i * pixels_per_image,
                cfg.width,
                cfg.height);
    }
    auto end = std::chrono::high_resolution_clock::now();
    t.cpu_blur_ms = std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.num_images; ++i) {
        CpuSobel(blurred.data() + i * pixels_per_image,
                 edges.data() + i * pixels_per_image,
                 cfg.width,
                 cfg.height);
    }
    end = std::chrono::high_resolution_clock::now();
    t.cpu_sobel_ms = std::chrono::duration<double, std::milli>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.num_images; ++i) {
        CpuThreshold(edges.data() + i * pixels_per_image,
                     binary.data() + i * pixels_per_image,
                     cfg.width,
                     cfg.height,
                     cfg.threshold);
    }
    end = std::chrono::high_resolution_clock::now();
    t.cpu_threshold_ms = std::chrono::duration<double, std::milli>(end - start).count();

    auto cpu_end_total = std::chrono::high_resolution_clock::now();
    t.cpu_total_ms =
        std::chrono::duration<double, std::milli>(cpu_end_total - cpu_start_total).count();
    return t;
}

void RunGpuPipeline(const Config& cfg,
                    const std::vector<uint8_t>& input,
                    std::vector<float>& blurred_out,
                    std::vector<float>& edges_out,
                    std::vector<uint8_t>& binary_out,
                    Timings* t) {
    const size_t pixels_per_image = static_cast<size_t>(cfg.width) * cfg.height;
    const size_t total_pixels = static_cast<size_t>(cfg.num_images) * pixels_per_image;
    const size_t input_bytes = total_pixels * sizeof(uint8_t);
    const size_t float_bytes = total_pixels * sizeof(float);

    uint8_t* d_input = nullptr;
    float* d_blur = nullptr;
    float* d_edges = nullptr;
    uint8_t* d_binary = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_blur, float_bytes));
    CHECK_CUDA(cudaMalloc(&d_edges, float_bytes));
    CHECK_CUDA(cudaMalloc(&d_binary, input_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

    const dim3 block(kTileDim, kTileDim);
    const dim3 grid((cfg.width + block.x - 1) / block.x, (cfg.height + block.y - 1) / block.y);
    const int threads_1d = 256;

    cudaEvent_t total_start, total_stop, blur_start, blur_stop, sobel_start, sobel_stop,
        threshold_start, threshold_stop;
    CHECK_CUDA(cudaEventCreate(&total_start));
    CHECK_CUDA(cudaEventCreate(&total_stop));
    CHECK_CUDA(cudaEventCreate(&blur_start));
    CHECK_CUDA(cudaEventCreate(&blur_stop));
    CHECK_CUDA(cudaEventCreate(&sobel_start));
    CHECK_CUDA(cudaEventCreate(&sobel_stop));
    CHECK_CUDA(cudaEventCreate(&threshold_start));
    CHECK_CUDA(cudaEventCreate(&threshold_stop));

    CHECK_CUDA(cudaEventRecord(total_start));

    CHECK_CUDA(cudaEventRecord(blur_start));
    for (int i = 0; i < cfg.num_images; ++i) {
        GaussianBlurKernel<<<grid, block>>>(d_input + i * pixels_per_image,
                                            d_blur + i * pixels_per_image,
                                            cfg.width,
                                            cfg.height);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(blur_stop));

    CHECK_CUDA(cudaEventRecord(sobel_start));
    for (int i = 0; i < cfg.num_images; ++i) {
        SobelKernel<<<grid, block>>>(d_blur + i * pixels_per_image,
                                     d_edges + i * pixels_per_image,
                                     cfg.width,
                                     cfg.height);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(sobel_stop));

    CHECK_CUDA(cudaEventRecord(threshold_start));
    const int blocks_1d = static_cast<int>((total_pixels + threads_1d - 1) / threads_1d);
    ThresholdKernel<<<blocks_1d, threads_1d>>>(d_edges, d_binary, static_cast<int>(total_pixels), cfg.threshold);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(threshold_stop));

    CHECK_CUDA(cudaEventRecord(total_stop));
    CHECK_CUDA(cudaEventSynchronize(total_stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, blur_start, blur_stop));
    t->gpu_blur_ms = ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, sobel_start, sobel_stop));
    t->gpu_sobel_ms = ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, threshold_start, threshold_stop));
    t->gpu_threshold_ms = ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, total_start, total_stop));
    t->gpu_total_ms = ms;

    CHECK_CUDA(cudaMemcpy(blurred_out.data(), d_blur, float_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(edges_out.data(), d_edges, float_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(binary_out.data(), d_binary, input_bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(total_start));
    CHECK_CUDA(cudaEventDestroy(total_stop));
    CHECK_CUDA(cudaEventDestroy(blur_start));
    CHECK_CUDA(cudaEventDestroy(blur_stop));
    CHECK_CUDA(cudaEventDestroy(sobel_start));
    CHECK_CUDA(cudaEventDestroy(sobel_stop));
    CHECK_CUDA(cudaEventDestroy(threshold_start));
    CHECK_CUDA(cudaEventDestroy(threshold_stop));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_blur));
    CHECK_CUDA(cudaFree(d_edges));
    CHECK_CUDA(cudaFree(d_binary));
}

std::vector<uint8_t> FloatImageToByte(const float* data, size_t count) {
    float max_val = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        max_val = std::max(max_val, data[i]);
    }
    const float scale = max_val > 0.0f ? 255.0f / max_val : 1.0f;

    std::vector<uint8_t> out(count);
    for (size_t i = 0; i < count; ++i) {
        const int value = static_cast<int>(data[i] * scale);
        out[i] = static_cast<uint8_t>(ClampInt(value, 0, 255));
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = ParseArgs(argc, argv);
        fs::create_directories(cfg.output_dir);

        int device = 0;
        CHECK_CUDA(cudaSetDevice(device));
        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

        const size_t pixels_per_image = static_cast<size_t>(cfg.width) * cfg.height;
        const size_t total_pixels = static_cast<size_t>(cfg.num_images) * pixels_per_image;

        std::cout << "Generating " << cfg.num_images << " synthetic images of size "
                  << cfg.width << 'x' << cfg.height << "..." << std::endl;
        std::vector<uint8_t> input = GenerateSyntheticBatch(cfg);

        std::vector<float> cpu_blur(total_pixels, 0.0f);
        std::vector<float> cpu_edges(total_pixels, 0.0f);
        std::vector<uint8_t> cpu_binary(total_pixels, 0);

        std::vector<float> gpu_blur(total_pixels, 0.0f);
        std::vector<float> gpu_edges(total_pixels, 0.0f);
        std::vector<uint8_t> gpu_binary(total_pixels, 0);

        std::cout << "Running CPU pipeline..." << std::endl;
        Timings timings = RunCpuPipeline(cfg, input, cpu_blur, cpu_edges, cpu_binary);

        std::cout << "Running GPU pipeline on: " << prop.name << std::endl;
        RunGpuPipeline(cfg, input, gpu_blur, gpu_edges, gpu_binary, &timings);

        const size_t sample_idx0 = 0;
        const size_t sample_idx1 = std::min<size_t>(1, static_cast<size_t>(cfg.num_images - 1));
        const size_t offset0 = sample_idx0 * pixels_per_image;
        const size_t offset1 = sample_idx1 * pixels_per_image;

        SavePgm(cfg.output_dir + "/input_0.pgm", input.data() + offset0, cfg.width, cfg.height);
        SavePgm(cfg.output_dir + "/gpu_binary_0.pgm", gpu_binary.data() + offset0, cfg.width, cfg.height);
        SavePgm(cfg.output_dir + "/gpu_binary_1.pgm", gpu_binary.data() + offset1, cfg.width, cfg.height);

        const std::vector<uint8_t> gpu_blur_0 = FloatImageToByte(gpu_blur.data() + offset0, pixels_per_image);
        const std::vector<uint8_t> gpu_edges_0 = FloatImageToByte(gpu_edges.data() + offset0, pixels_per_image);
        SavePgm(cfg.output_dir + "/gpu_blur_0.pgm", gpu_blur_0.data(), cfg.width, cfg.height);
        SavePgm(cfg.output_dir + "/gpu_edges_0.pgm", gpu_edges_0.data(), cfg.width, cfg.height);

        if (cfg.save_cpu_outputs) {
            const std::vector<uint8_t> cpu_blur_0 = FloatImageToByte(cpu_blur.data() + offset0, pixels_per_image);
            const std::vector<uint8_t> cpu_edges_0 = FloatImageToByte(cpu_edges.data() + offset0, pixels_per_image);
            SavePgm(cfg.output_dir + "/cpu_blur_0.pgm", cpu_blur_0.data(), cfg.width, cfg.height);
            SavePgm(cfg.output_dir + "/cpu_edges_0.pgm", cpu_edges_0.data(), cfg.width, cfg.height);
            SavePgm(cfg.output_dir + "/cpu_binary_0.pgm", cpu_binary.data() + offset0, cfg.width, cfg.height);
        }

        WriteTimingCsv(cfg.output_dir + "/timings.csv", timings);
        WriteExecutionLog(cfg.output_dir + "/execution_log.txt", cfg, timings, prop);

        const double speedup = timings.gpu_total_ms > 0.0 ? timings.cpu_total_ms / timings.gpu_total_ms : 0.0;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "CPU total time: " << timings.cpu_total_ms << " ms\n";
        std::cout << "GPU total time: " << timings.gpu_total_ms << " ms\n";
        std::cout << "Overall speedup: " << speedup << "x\n";
        std::cout << "Saved outputs to: " << cfg.output_dir << std::endl;

        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
