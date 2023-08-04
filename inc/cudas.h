//
// Created by yuanlu on 2023/8/4.
//

#ifndef NUEDC_2023_CUDAS_H
#define NUEDC_2023_CUDAS_H

#include <opencv2/core/cuda.hpp>

namespace ifr::cuda {
    void findPointer(const cv::cuda::GpuMat &src, const cv::cuda::GpuMat *mask, cv::cuda::GpuMat &dst,
                     int t1 = 30, int t2 = 230);

    void inv(const cv::cuda::GpuMat &img);

    void keepFill(const cv::cuda::GpuMat &img, const cv::Rect &rect, int padding = 2);
}
#endif //NUEDC_2023_CUDAS_H
