//
// Created by yuanlu on 2022/9/7.
//

#include "Camera.h"
#include <cuda_runtime_api.h>
#include "opencv2/opencv.hpp"
#include "mem_map.h"
#include "logger.h"


using namespace std::placeholders;

namespace ifr {

    void Camera::runCamera() const {
        //发送开采命令
        IFR_GX_CHECK(GXStreamOn(m_hDevice));
    }

    void Camera::initCamera() {
        GX_OPEN_PARAM openParam;
        uint32_t nDeviceNum = 0;
        openParam.accessMode = GX_ACCESS_EXCLUSIVE;
        openParam.openMode = GX_OPEN_INDEX;
        openParam.pszContent = new char[2]{'1', '\0'};
        // 初始化库
        IFR_GX_CHECK(GXInitLib());

        // 枚举设备列表
        IFR_GX_CHECK(GXUpdateDeviceList(&nDeviceNum, 1000));
        IFR_GX_ASSERT(nDeviceNum);

        //打开设备
        IFR_GX_CHECK(GXOpenDevice(&openParam, &m_hDevice));

        if (use_trigger) {
            GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_ON);
            GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_SWITCH, GX_TRIGGER_SWITCH_ON);
            GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_SOURCE, GX_TRIGGER_SOURCE_LINE2);
            GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_ACTIVATION, GX_TRIGGER_ACTIVATION_RISINGEDGE);
            GXSetFloat(m_hDevice, GX_FLOAT_TRIGGER_FILTER_RAISING, 0.0f);
            GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_SELECTOR, GX_ENUM_TRIGGER_SELECTOR_FRAME_START);
        } else {
            GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_SWITCH, GX_TRIGGER_SWITCH_OFF);
            GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);
        }

        if (fps > 0) {
            GXSetFloat(m_hDevice, GX_FLOAT_ACQUISITION_FRAME_RATE, fps);
            GXSetEnum(m_hDevice, GX_ENUM_ACQUISITION_FRAME_RATE_MODE, GX_ACQUISITION_FRAME_RATE_MODE_ON);
        } else {
            GXSetEnum(m_hDevice, GX_ENUM_ACQUISITION_FRAME_RATE_MODE, GX_ACQUISITION_FRAME_RATE_MODE_OFF);
        }

        //设置采集模式连续采集
        GXSetEnum(m_hDevice, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
        GXSetInt(m_hDevice, GX_INT_ACQUISITION_SPEED_LEVEL, 1);
        GXSetEnum(m_hDevice, GX_ENUM_BALANCE_WHITE_AUTO, GX_BALANCE_WHITE_AUTO_CONTINUOUS);
        //设置自动曝光
        IFR_GX_CHECK(GXSetEnum(m_hDevice, GX_ENUM_EXPOSURE_AUTO, GX_EXPOSURE_AUTO_OFF));    //关闭自动曝光
        IFR_GX_CHECK(GXSetFloat(m_hDevice, GX_FLOAT_EXPOSURE_TIME, exposure)); //初始曝光时间
        //IFR_GX_CHECK(GXSetEnum(m_hDevice, GX_ENUM_EXPOSURE_AUTO, GX_EXPOSURE_AUTO_CONTINUOUS));//自动曝光
        //IFR_GX_CHECK(GXSetFloat(m_hDevice, GX_FLOAT_AUTO_EXPOSURE_TIME_MIN, 20.0000));//自动曝光最小值
        //IFR_GX_CHECK(GXSetFloat(m_hDevice, GX_FLOAT_AUTO_EXPOSURE_TIME_MAX, 100000.0000));//自动曝光最大值
        //IFR_GX_CHECK(GXSetInt(m_hDevice, GX_INT_GRAY_VALUE, 150));//期望灰度值
        //设置自动白平衡
        //GXSetEnum(m_hDevice, GX_ENUM_BALANCE_WHITE_AUTO, GX_BALANCE_WHITE_AUTO_CONTINUOUS); //自动白平衡
        //GXSetEnum(m_hDevice, GX_ENUM_AWB_LAMP_HOUSE, GX_AWB_LAMP_HOUSE_ADAPTIVE); //自动白平衡光源

        //bool      bColorFliter = false;
        GXGetInt(m_hDevice, GX_INT_PAYLOAD_SIZE, &m_nPayLoadSize);  // 获取图像大小
        GXGetInt(m_hDevice, GX_INT_WIDTH, &m_nImageWidth); // 获取宽度
        GXGetInt(m_hDevice, GX_INT_HEIGHT, &m_nImageHeight);// 获取高度
        GXGetEnum(m_hDevice, GX_ENUM_PIXEL_SIZE, &m_nPixelSize);
        //IFR_GX_CHECK(GXSetFloat(m_hDevice, GX_FLOAT_EXPOSURE_TIME, 20000));


//                cudaHostAlloc(reinterpret_cast<void **>(&host_ptr), data_size,
//                              cudaHostAllocMapped | cudaHostAllocWriteCombined);
        uint8_t pBuffer;
        size_t pnSize = 0;
        GXGetBufferLength(m_hDevice, GX_BUFFER_FRAME_INFORMATION, &pnSize);
        std::cout << "Buffer Length = " << pnSize << ", payLoadSize = " << m_nPayLoadSize << std::endl;
        GXGetBuffer(m_hDevice, GX_FEATURE_ID::GX_BUFFER_FRAME_INFORMATION, &pBuffer, &pnSize);

        IFR_GX_ASSERT(m_nImageHeight > 0 && m_nImageWidth > 0);

        //判断相机是否支持bayer格式
        bool m_bColorFilter;
        IFR_GX_CHECK(GXIsImplemented(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_bColorFilter));
        if (m_bColorFilter) {
            IFR_GX_CHECK(GXGetEnum(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_nPixelColorFilter));
        }
    }

    void Camera::stopCamera() const {
        //发送停采命令
        IFR_GX_CHECK(GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP));
        //注销采集回调

        IFR_GX_CHECK(GXStreamOff(m_hDevice));

        IFR_GX_CHECK(GXCloseDevice(m_hDevice));
        IFR_GX_CHECK(GXCloseLib());
    }


    int64_t Camera::getHeight() const { return m_nImageHeight; }

    int64_t Camera::getWidth() const { return m_nImageWidth; }


} // ifr