#include "cudas.h"

namespace ifr::cuda {

    __global__ void k_findPointer(const cv::cuda::PtrStepSz<uint8_t> src,
                                  const cv::cuda::PtrStepSz<uint8_t> mask,
                                  cv::cuda::PtrStepSz<uint8_t> dst,
                                  uint8_t t1, uint8_t t2) {
        int i = (int(threadIdx.x) + blockIdx.x * blockDim.x);
        int j = (int(threadIdx.y) + blockIdx.y * blockDim.y);
        if (i >= src.cols || j >= src.rows) return;

        dst(j, i) = (src(j, i) > (mask(j, i) ? t1 : t2)) ? 255 : 0;

    }

    __global__ void k_findPointer(const cv::cuda::PtrStepSz<uint8_t> src,
                                  cv::cuda::PtrStepSz<uint8_t> dst,
                                  uint8_t t) {
        int i = (int(threadIdx.x) + blockIdx.x * blockDim.x);
        int j = (int(threadIdx.y) + blockIdx.y * blockDim.y);
        if (i >= src.cols || j >= src.rows) return;

        dst(j, i) = (src(j, i) > t) ? 255 : 0;

    }

    __global__ void k_inv(cv::cuda::PtrStepSz<uint8_t> img) {
        int i = (int(threadIdx.x) + blockIdx.x * blockDim.x);
        int j = (int(threadIdx.y) + blockIdx.y * blockDim.y);
        if (i >= img.cols || j >= img.rows) return;

        auto &v = img(j, i);
        v = 255 - v;
    }

    __global__ void k_keepFill(cv::cuda::PtrStepSz<uint8_t> img, int x1, int y1, int x2, int y2) {
        int i = (int(threadIdx.x) + blockIdx.x * blockDim.x);
        int j = (int(threadIdx.y) + blockIdx.y * blockDim.y);
        if (i >= img.cols || j >= img.rows) return;
        if (x1 <= i && i <= x2) return;
        if (y1 <= j && j <= y2) return;
        img(j, i) = 0;
    }

    void findPointer(const cv::cuda::GpuMat &src, const cv::cuda::GpuMat *mask, cv::cuda::GpuMat &dst,
                     int t1, int t2) {
        dim3 blockDim(32, 32);
        dim3 gridDim((src.cols + blockDim.x - 1) / blockDim.x, (src.rows + blockDim.y - 1) / blockDim.y);
        if (mask == nullptr)
            k_findPointer<<<gridDim, blockDim, 0>>>(src, dst, t2);
        else
            k_findPointer<<<gridDim, blockDim, 0>>>(src, *mask, dst, t1, t2);
    }

    void inv(const cv::cuda::GpuMat &img) {
        dim3 blockDim(32, 32);
        dim3 gridDim((img.cols + blockDim.x - 1) / blockDim.x, (img.rows + blockDim.y - 1) / blockDim.y);
        k_inv<<<gridDim, blockDim, 0>>>(img);
    }

    void keepFill(const cv::cuda::GpuMat &img, const cv::Rect &rect, int padding) {
        dim3 blockDim(32, 32);
        dim3 gridDim((img.cols + blockDim.x - 1) / blockDim.x, (img.rows + blockDim.y - 1) / blockDim.y);
        auto x1 = std::max(rect.x - padding, 0);
        auto y1 = std::max(rect.y - padding, 0);
        int x2 = std::min(rect.x + rect.width + padding, img.cols - 1);
        auto y2 = std::min(rect.y + rect.height + padding, img.rows - 1);
        k_keepFill<<<gridDim, blockDim, 0>>>(img, x1, y1, x2, y2);

    }
}