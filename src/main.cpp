//
// Created by yuanlu on 2023/4/14.
//
#include "opencv2/opencv.hpp"
#include "./common.h"
#include "./Camera.h"
#include "utility"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "opencv2/cudafilters.hpp"
#include "cudas.h"
#include "serial/serial.h"
#include "pkg.h"


static_assert(__OS__ == __OS_Linux__);


#define TIME 0
#define DRAW 0

#if TIME
#define REC_T(t) const auto ___time_rec___##t = cv::getTickCount()
#define PRINT_T(t1, t2) std::cout<< __func__ << " time: " #t1 "~" #t2 " = " << (double(___time_rec___##t2 - ___time_rec___##t1) / cv::getTickFrequency() * 1000) << " ms" << std::endl
#else
#define REC_T(t) do{}while(0)
#define PRINT_T(t1, t2) do{}while(0)
#endif
#if DRAW

std::string createWindow(const std::string &name) {

    cv::namedWindow(name, cv::WINDOW_NORMAL);
    return name;
}

#define SHOW(name, img) do{static const auto name_1=createWindow(name);cv::imshow(name_1, img);}while(0)
#else
#define SHOW(name, img) do{}while(0)
#endif


class Handler {
#define PAIR_MAT(name) cv::Mat c_##name;cv::cuda::GpuMat g_##name
#define PM_MALLOC(name)   memMapPool.malloc(c_##name, g_##name, src.size(), src.type())
public:
    cv::Point2d green, red;//激光位置
    std::vector<cv::Point> path;//路径
private:
    int step = -1;
    cv::Point2d end;
    serial::Serial *serial = nullptr;
    ifr::pkg::Move pkg_move;

    const double min_distance = std::pow(20.0, 2);


    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10));
    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kernel);
private:
#if DRAW

    FORCE_INLINE void
    drawRotatedRect(cv::Mat &mat, const cv::RotatedRect &rr, const cv::Scalar &color, int thickness = 1,
                    int lineType = cv::LINE_8) {
        cv::Point2f vertices[4];
        rr.points(vertices);

        // 在图像上绘制旋转矩形
        for (int i = 0; i < 4; ++i) {
            cv::line(mat, vertices[i], vertices[(i + 1) % 4], color, thickness, lineType);
        }

    }

    FORCE_INLINE void
    drawPoly(cv::Mat &mat, const std::vector<cv::Point> &pts, const cv::Scalar &color, int thickness = 1,
             int lineType = cv::LINE_8) {
        const auto size = pts.size();
        // 在图像上绘制旋转矩形
        for (std::size_t i = 0; i < size; ++i) {
            cv::line(mat, pts[i], pts[(i + 1) % size], color, thickness, lineType);
        }

    }

#endif


    PAIR_MAT(gray);
    PAIR_MAT(otsu);
    PAIR_MAT(bin);
    bool has_bin = false;

    inline auto findArea(rm_armor_finder::MemMapPool &memMapPool, const cv::Mat &src, const cv::cuda::GpuMat &gpu) {
        PM_MALLOC(gray);
        PM_MALLOC(otsu);
        PM_MALLOC(bin);
        REC_T(0);
        cv::cuda::cvtColor(gpu, g_gray, cv::COLOR_BGR2GRAY);
        REC_T(1);
        cv::threshold(c_gray, c_otsu, 30, 255, cv::THRESH_OTSU);
        REC_T(2);
        filter->apply(g_otsu, g_bin);


        REC_T(3);
        ifr::cuda::inv(g_bin);

        REC_T(4);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(c_bin, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);


        REC_T(5);
#if DRAW
        cv::Mat draw;
        cv::cvtColor(c_bin, draw, cv::COLOR_GRAY2BGR);
#endif

        std::vector<cv::Point> max_approx, max_parent_approx, parent_approx, child_approx, mid_approx{4};
        double max_size = -1;
        for (size_t i = 0; i < contours.size(); i++) {
            auto child = hierarchy[i][2];
            if (child < 0)continue;//没有子轮廓


            const auto &contour = contours[i];
            {
                double epsilon = 0.04 * cv::arcLength(contour, true);
                cv::approxPolyDP(contour, parent_approx, epsilon, true);
                if (parent_approx.size() != 4)continue;
            }
            do {
                const auto &child_contour = contours[child];
                double epsilon = 0.04 * cv::arcLength(child_contour, true);
                cv::approxPolyDP(child_contour, child_approx, epsilon, true);
                if (child_approx.size() != 4)continue;
                double min_dis, max_dis;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    double dis;
                    int min_k = 0;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        auto vec = parent_approx[j] - child_approx[k];
                        auto dis0 = vec.ddot(vec);
                        if (k == 0 || dis0 < dis) {
                            dis = dis0;
                            min_k = k;
                        }
                    }
                    mid_approx[j] = (parent_approx[j] + child_approx[min_k]) / 2;
                    if (j == 0 || dis < min_dis)min_dis = dis;
                    if (j == 0 || dis > max_dis)max_dis = dis;
                }
                if (std::sqrt(max_dis) / std::sqrt(min_dis) < 3)break;//矩形贴合
            } while ((child = hierarchy[child][0]) >= 0);

#if DRAW
            drawPoly(draw, parent_approx, {255, 255, 0}, 2, cv::LINE_AA);
#endif
            if (child < 0)continue;
#if DRAW
            drawPoly(draw, child_approx, {0, 255, 0}, 2, cv::LINE_AA);
            drawPoly(draw, mid_approx, {0, 255, 255}, 2, cv::LINE_AA);
#endif

            auto size = cv::contourArea(contour);
            if (i == 0 || size > max_size) {
                max_size = size;
                max_approx = mid_approx;
                max_parent_approx = parent_approx;
            }

        }

        REC_T(6);

        if (max_size > 0) {
            ifr::cuda::keepFill(g_bin, cv::boundingRect(max_parent_approx));
            has_bin = true;
        } else {
            has_bin = false;
        }
        path = max_approx;
        if (!max_approx.empty()) path.push_back(max_approx[0]);

        SHOW("area", draw);

        REC_T(7);

        PRINT_T(0, 1);
        PRINT_T(1, 2);
        PRINT_T(2, 3);
        PRINT_T(3, 4);
        PRINT_T(4, 5);
        PRINT_T(5, 6);
        PRINT_T(6, 7);
        PRINT_T(0, 7);

    }

    PAIR_MAT(hsv);
    PAIR_MAT(HSV[3]);
    PAIR_MAT(pointer);

    inline auto findPointer(rm_armor_finder::MemMapPool &memMapPool, const cv::Mat &src, const cv::cuda::GpuMat &gpu) {
        PM_MALLOC(hsv);
#pragma unroll
        for (int i = 0; i < 3; i++)
            PM_MALLOC(HSV[i]);
        PM_MALLOC(pointer);

        cv::cuda::cvtColor(gpu, g_hsv, cv::COLOR_BGR2HSV);
        cv::cuda::split(g_hsv, g_HSV);

        ifr::cuda::findPointer(g_HSV[2], has_bin ? &g_bin : nullptr, g_pointer);


        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(c_pointer, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

#if DRAW
        cv::Mat draw;
        cv::cvtColor(c_pointer, draw, cv::COLOR_GRAY2BGR);
#endif

        green = {-1, -1};
        red = {-1, -1};
        double size_green = 0, size_red = 0;
        for (const auto &contour: contours) {
            double area = cv::contourArea(contour);
            if (area > 300 || area < 5)continue;
            cv::Rect rect = cv::boundingRect(contour);

            double aspectRatio = static_cast<double>(rect.width) / rect.height;
            if (aspectRatio < 0.5 || aspectRatio > 2.0)continue;
            cv::Point2d center{rect.x + rect.width / 2.0, rect.y + rect.height / 2.0};
            const auto side = (std::max)(rect.width, rect.height);//最大边长

            int type = 0;
            const int fx = (std::max)(rect.x - side, 0), fy = (std::max)(rect.y - side, 0);
            const int tx = (std::min)(rect.x + rect.width + side, c_pointer.rows), ty = (std::min)(
                    rect.y + rect.height + side, c_pointer.cols);

            for (auto y = fy; y < ty; ++y) {
                for (auto x = fx; x < tx; ++x) {
                    if (c_pointer.at<uchar>(y, x))continue;//太亮了
                    const auto &v = c_HSV[2].at<uchar>(y, x);
                    if (v < 10)continue;//太暗了
                    const auto &s = c_HSV[1].at<uchar>(y, x);
                    if (s < 60)continue;//饱和度太低了(过滤白色)
                    const auto &h = c_HSV[0].at<uchar>(y, x);

                    //[black,red,green,blue,white]
                    //[{0,   0,   0}, {0, 255, 255}, {60, 255, 255}, {120, 255, 255}, {0,   0, 255}]
                    auto r = (std::min)(int(h), 180 - h);
                    auto g = std::abs(60 - h);
                    if (r < 10)--type;
                    else if (g < 10)++type;
                }
            }

            if (type > 3) {
                if (area > size_green) {
                    green = center;
                    size_green = area;
                }
            } else if (type < 3) {
                if (area > size_red) {
                    red = center;
                    size_red = area;
                }
            }
        }

#if DRAW
        if (green.x >= 0 && green.y >= 0)
            cv::circle(draw, green, 5, {0, 255, 0}, -1, cv::LINE_AA);
        if (red.x >= 0 && red.y >= 0)
            cv::circle(draw, red, 5, {0, 0, 255}, -1, cv::LINE_AA);
#endif

        SHOW("pointer", draw);

    }

public:
    Handler() {
        serial = new serial::Serial(
                "/dev/ttyUSB0",
                115200,
                serial::Timeout::simpleTimeout(500),
                static_cast<serial::bytesize_t>(8),
                static_cast<serial::parity_t>(0),
                static_cast<serial::stopbits_t>(1),
                static_cast<serial::flowcontrol_t>(0));
        IFR_LOG_STREAM("Serial", "serial open: " << serial->isOpen());
        if (!serial->isOpen())throw std::runtime_error("Can not open serial port");
    }

    void handler(rm_armor_finder::MemMapPool &memMapPool, cv::Mat &img, cv::cuda::GpuMat &gpu, bool flip = false) {
        REC_T(1);
        if (flip)
            cv::cuda::flip(gpu, gpu, -1);  // 使用负数表示旋转180度
        REC_T(2);
        cv::cuda::cvtColor(gpu, gpu, cv::COLOR_BayerRG2BGR);
        REC_T(3);
        findArea(memMapPool, img, gpu);
        REC_T(4);
        if (!path.empty()) findPointer(memMapPool, img, gpu);
        else red = green = {-1, -1};
        REC_T(5);

        PRINT_T(1, 2);
        PRINT_T(2, 3);
        PRINT_T(3, 4);
        PRINT_T(4, 5);
        PRINT_T(1, 5);
    }

    cv::Point2d move_step() {
        if (red.x < 0 || red.y < 0)return {0, 0};
        if (step < 0) {
            end = red;
            step = 0;
        }
        while (step < path.size()) {
            cv::Point2d go = path[step];
            go -= red;
            if (go.ddot(go) < min_distance)
                step++;
            else
                return go;
        }
        return end - red;
    }

    CV_NODISCARD_STD cv::Point2d follow() const {
        if (red.x < 0 || red.y < 0 || green.x < 0 || green.y < 0)return {};
        return green - red;
    }

    void do_green() {
        auto go = follow();
        pkg_move.set(go.x, go.y);
        IFR_LOG_STREAM("DO", go << ", red = " << red << ", green = " << green << ", path = " << path.size());
        serial->writeT(pkg_move);
    }

    bool do_red() {
        cv::Point2d go;
        bool finish = false;
        if (red.x < 0 || red.y < 0) {
        } else {
            go = move_step();
            finish = go.ddot(go) < min_distance;
            if (finish)step = -1;
        }
        pkg_move.set(go.x, go.y);
        serial->writeT(pkg_move);
        IFR_LOG_STREAM("DO",
                       go << ", red = " << red << ", green = " << green << ", path = " << path.size() << ", finish = "
                          << (finish ? "true" : "false") << ", step = " << step);
        return finish;
    }

private:
    std::string read_buf;
public:
    /**
     * 尝试读取字符串, 并将maps中对应值设为true
     * @param maps 关键词到对应值的映射
     */
    void tryRead(std::unordered_map<std::string, bool *> &maps) {
        auto size = serial->available();
        if (size > 0) {
            read_buf += serial->read(size);
            size_t pos = -1;
            while ((pos = read_buf.find('\n')) != std::string::npos) {
                size_t start = 0;
                while (start < pos && read_buf[start] == '\0') start++;
                auto line = read_buf.substr(start, pos - start);
                read_buf = read_buf.substr(pos + 1);
                if (!line.empty()) {
                    auto &ptr = maps[line];
                    if (ptr != nullptr && !(*ptr)) {
                        IFR_LOG_STREAM("serial", "成功读取: " << line);
                        *ptr = true;
                    }
                }
            }
            if (read_buf.size() > 256)read_buf = "";
        }
    }
};


void run() {
    ifr::Camera camera(2000, false, 100);
    camera.initCamera();
    camera.runCamera();

    Handler handler;

    bool is_red = false, is_green = false, is_stop = false;
    std::unordered_map<std::string, bool *> keyword = {
            {"1_ihw9jnsh39m", &is_green},
            {"2_9kitey3yzpd", &is_red},
            {"3_yp4lmg19kbc", &is_stop},
    };
    while (true) {
        PGX_FRAME_BUFFER pFrameBuffer;

        IFR_GX_CHECK(GXDQBuf(camera.m_hDevice, &pFrameBuffer, 500));
        if (pFrameBuffer->nStatus == GX_FRAME_STATUS_SUCCESS) {
            cv::Mat src(pFrameBuffer->nHeight, pFrameBuffer->nWidth, CV_8UC1,
                        reinterpret_cast<void *>(pFrameBuffer->pImgBuf));
            void *cuda_src_ptr = camera.memMapPool.getDevicePointer(src.data, src.dataend - src.datastart);
            cv::cuda::GpuMat gpu_src(src.rows, src.cols, src.type(), cuda_src_ptr);

            handler.handler(camera.memMapPool, src, gpu_src, true);

            handler.tryRead(keyword);
            if ((is_red && is_green) || is_stop)is_red = is_green = is_stop = false;
            if (is_red) {
                if (handler.do_red())is_red = false;
            }
            if (is_green) {
                handler.do_green();
            }
//            IFR_LOG_STREAM("Main", handler.path);
        }
        IFR_GX_CHECK(GXQBuf(camera.m_hDevice, pFrameBuffer));

        SLEEP(SLEEP_TIME(10.0 / 1000));
//        cv::waitKey(10);
    }
}


int main() {
    ifr::logger::setMainPath(__FILE__);
//    {
//        cv::Mat x(cv::Size{5, 1}, CV_8UC3, {0, 0, 0});
//        x.at<cv::Vec3b>(0, 0) = {0, 0, 0};
//        x.at<cv::Vec3b>(0, 1) = {255, 0, 0};
//        x.at<cv::Vec3b>(0, 2) = {0, 255, 0};
//        x.at<cv::Vec3b>(0, 3) = {0, 0, 255};
//        x.at<cv::Vec3b>(0, 4) = {255, 255, 255};
//        std::cout << x << std::endl;
//        cv::cvtColor(x, x, cv::COLOR_RGB2HSV);
//        std::cout << x << std::endl;
//    }
//
//    return 0;
    run();

    return 0;
}