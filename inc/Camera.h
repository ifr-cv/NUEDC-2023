//
// Created by yuanlu on 2022/9/7.
//

#ifndef IFR_OPENCV_CAMERA_H
#define IFR_OPENCV_CAMERA_H

#include "GxIAPI.h"
#include "DxImageProc.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>
#include "mem_map.h"

namespace ifr {

#define IFR_GX_ERROR(expr, stat) do{IFR_LOG_STREAM("Camera",#expr ": "<<__stat__);throw std::runtime_error(#expr); }while(0)
#define IFR_GX_CHECK(expr) do{auto __stat__=expr;if (__stat__!=GX_STATUS_SUCCESS) IFR_GX_ERROR(expr,__stat__);}while(0)
#define IFR_GX_ASSERT(expr) do{auto __stat__=expr;if (!__stat__) IFR_GX_ERROR(expr,__stat__);}while(0)

    /**
     * 代表相机接口
     */
    class Camera {
    public:
        GX_DEV_HANDLE m_hDevice = {};        //设备句柄
        int64_t m_nImageHeight = {};         //原始图像高
        int64_t m_nImageWidth = {};          //原始图像宽
        int64_t m_nPayLoadSize = {};         //数据大小
        int64_t m_nPixelColorFilter = {};    //Bayer格式
        int64_t m_nPixelSize = {};           //像素深度
        rm_armor_finder::MemMapPool memMapPool;


        const float exposure;//曝光时长
        const bool use_trigger;//是否使用硬触发
        const float fps;//帧率

        Camera(float exposure, bool use_trigger, float fps) :
                exposure(exposure), use_trigger(use_trigger), fps(fps) {}

        ~Camera() { stopCamera(); }

        /**启动相机*/
        void runCamera() const;

        void initCamera();

        /**停止相机*/
        void stopCamera() const;

        /**@return 图像高度*/
        CV_NODISCARD_STD int64_t getHeight() const;

        /**@return 图像宽度*/
        CV_NODISCARD_STD int64_t getWidth() const;

    };


} // ifr

#endif //IFR_OPENCV_CAMERA_H
