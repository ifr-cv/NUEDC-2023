#ifndef IFR_ROS2_CV_PACKAGE_RM_ARMOR_FINDER_MEM_MAP_H
#define IFR_ROS2_CV_PACKAGE_RM_ARMOR_FINDER_MEM_MAP_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "logger.h"

namespace rm_armor_finder {
    class MemMapPool {
#define RAF_MMP_LOG(...)  IFR_LOG_STREAM("MemMapPool" , __VA_ARGS__)
#define RAF_MMP_LOG_CUDA(expr, err)                                                                          \
    do {                                                                                                     \
        auto result = expr;                                                                                  \
        static_assert(std::is_same_v<decltype(result), decltype(cudaError::cudaSuccess)>, "Need Cuda expr"); \
        if (result == cudaError::cudaSuccess) {                                                              \
            if constexpr (DEBUG) RAF_MMP_LOG((#expr " = ")<< result);                                     \
        } else                                                                                               \
            RAF_MMP_LOG((#expr " = ") << result                                                              \
                                             << ", " << err                                                  \
                                             << " (" << cudaGetErrorName(result) << ")"                      \
                                             << "->" << cudaGetErrorString(result));                         \
    } while (0)
        std::unordered_map<intptr_t, void *> mem;///< CPU内存到GPU内存的映射

        static inline intptr_t toKey(const void *const ptr) { return reinterpret_cast<intptr_t>(ptr); }

    public:

        ~MemMapPool() {
            for (const auto &x: mem) {
                RAF_MMP_LOG("Unregister host memory " << x.second);
                cudaHostUnregister(x.second);
            }
        }


        ///@brief 通过主机地址得到设备地址
        void *getDevicePointer(void *host_ptr, size_t size, bool readonly = false) {
            auto &dev_ptr = mem[toKey(host_ptr)];
            if (dev_ptr != nullptr) [[likely]] {
                return dev_ptr;
            } else {
                uint flag = cudaHostRegisterMapped | cudaHostRegisterPortable;
                if (readonly) flag |= cudaHostRegisterReadOnly;
                //else flag |= cudaHostRegisterIoMemory;
                RAF_MMP_LOG_CUDA(cudaHostRegister(host_ptr, size, flag), "Can NOT Reg Host Memory.");
                RAF_MMP_LOG_CUDA(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0), "Can NOT Map Host Memory.");
                if constexpr (DEBUG) RAF_MMP_LOG("Get device memory " << dev_ptr);
                return dev_ptr;
            }

        }

        ///@brief 分配CPU与GPU共享的内存
        void malloc(cv::Mat &cpu, cv::cuda::GpuMat &gpu, cv::Size size, int type) {
            if (cpu.empty()) [[unlikely]] {
                cpu.create(size, type);
                gpu = cv::cuda::GpuMat(size, type, getDevicePointer(cpu.data, cpu.dataend - cpu.datastart));
            }
            CV_DbgAssert(cpu.size() == size && cpu.type() == type);
            CV_DbgAssert(gpu.size() == size && gpu.type() == type);
        }

#undef RAF_MMP_LOG
#undef RAF_MMP_LOG_CUDA
    };
}// namespace rm_armor_finder

#endif// IFR_ROS2_CV_PACKAGE_RM_ARMOR_FINDER_MEM_MAP_H
