//
// Created by yuanlu on 2022/11/8.
//

#ifndef COMMON_MODULES_LOGGER_H
#define COMMON_MODULES_LOGGER_H

#include "string"
#include "iostream"
#include "mutex"

namespace ifr {
    namespace logger {

#define IFR_LOGGER(type, __Expr__) ifr::logger::log(type,#__Expr__,__Expr__) //打印表达式字符串及其结果
#define IFR_LOGGER_LOE(type, __Expr__) ifr::logger::log_or_err(type,#__Expr__,__Expr__)//打印表达式及其结果(若结果为false(或0等)则使用err打印)
#define IFR_LOC_LOGGER(__Expr__) ifr::logger::log_loc(__FILE__, __LINE__, __func__,#__Expr__,__Expr__)//定位打印
#define IFR_LOG_STREAM(type, ...) do{ \
        std::unique_lock<std::mutex> lock(ifr::logger::getLogMtx());\
        ifr::logger::print_time(std::cout);\
        std::cout << ("[" type "] ") << __VA_ARGS__<<std::endl;\
}while(0)

        /**
         * @brief 输出当前时间字符串
         * @details [HH:MM:SS]
         */
        void print_time(std::ostream &outer);

        ///获取锁
        std::mutex &getLogMtx();

        /**
         * @brief 获取文件名
         * @param str 绝对路径
         * @return 相对路径
         */
        std::string getFile(const std::string &str);

        ///设置主文件的路径
        void setMainPath(const std::string &main);

        /**
         * @brief 打印一行日志
         * @details 4种输出形式
         * @details [type id] sub_type data
         * @details [type] sub_type data
         * @details [type id] data
         * @details [type] data
         * @tparam T log内容类型
         * @param type log分类
         * @param sub_type log子分类(为空不输出)
         * @param data log内容
         * @param id 分类ID(小于0不输出)
         */
        template<typename T>
        inline void log(const std::string &type, const std::string &sub_type, const T &data) {
            std::unique_lock<std::mutex> lock(getLogMtx());
            print_time(std::cout);
            std::cout << '[' << type;
            std::cout << ']';
            if (!sub_type.empty())std::cout << ' ' << sub_type << ':';
            std::cout << ' ' << data << std::endl;
        }

        /**
         * @brief 打印一行日志
         * @details 2种输出形式
         * @details [type id] data
         * @details [type] data
         * @tparam T log内容类型
         * @param type log分类
         * @param data log内容
         * @param id 分类ID(小于0不输出)
         */
        template<typename T>
        inline void log(const std::string &type, const T &data) { log(type, "", data); }

        /**
         * @brief 打印一行日志
         * @details 4种输出形式
         * @details [type id] sub_type data
         * @details [type] sub_type data
         * @details [type id] data
         * @details [type] data
         * @tparam T log内容类型
         * @param type log分类
         * @param id 分类ID(小于0不输出)
         * @param sub_type log子分类(为空不输出)
         * @param data log内容
         */
        template<typename T>
        inline void log(const std::string &type, int id, const std::string &sub_type, const T &data) {
            std::unique_lock<std::mutex> lock(getLogMtx());
            print_time(std::cout);
            std::cout << '[' << type;
            if (id >= 0)std::cout << ' ' << id;
            std::cout << ']';
            if (!sub_type.empty())std::cout << ' ' << sub_type << ':';
            std::cout << ' ' << data << std::endl;
        }

        /**
         * @brief 打印一行日志
         * @details 2种输出形式
         * @details [type id] data
         * @details [type] data
         * @tparam T log内容类型
         * @param type log分类
         * @param id 分类ID(小于0不输出)
         * @param data log内容
         */
        template<typename T>
        inline void log(const std::string &type, int id, const T &data) { log(type, id, "", data); }


        /**
         * @brief 打印一行错误日志
         * @details 4种输出形式
         * @details [type id] sub_type data
         * @details [type] sub_type data
         * @details [type id] data
         * @details [type] data
         * @tparam T log内容类型
         * @param type log分类
         * @param sub_type log子分类(为空不输出)
         * @param data log内容
         * @param id 分类ID(小于0不输出)
         */
        template<typename T>
        inline void err(const std::string &type, const std::string &sub_type, const T &data) {
            std::unique_lock<std::mutex> lock(getLogMtx());
            print_time(std::cerr);
            std::cerr << '[' << type;
            std::cerr << ']';
            if (!sub_type.empty())std::cerr << ' ' << sub_type << ':';
            std::cerr << ' ' << data << std::endl;
        }

        /**
         * @brief 打印一行错误日志
         * @details 2种输出形式
         * @details [type id] data
         * @details [type] data
         * @tparam T log内容类型
         * @param type log分类
         * @param data log内容
         * @param id 分类ID(小于0不输出)
         */
        template<typename T>
        inline void err(const std::string &type, const T &data) { err(type, "", data); }

        /**
         * @brief 打印一行错误日志
         * @details 4种输出形式
         * @details [type id] sub_type data
         * @details [type] sub_type data
         * @details [type id] data
         * @details [type] data
         * @tparam T log内容类型
         * @param type log分类
         * @param id 分类ID(小于0不输出)
         * @param sub_type log子分类(为空不输出)
         * @param data log内容
         */
        template<typename T>
        inline void err(const std::string &type, int id, const std::string &sub_type, const T &data) {
            std::unique_lock<std::mutex> lock(getLogMtx());
            print_time(std::cerr);
            std::cerr << '[' << type;
            if (id >= 0)std::cerr << ' ' << id;
            std::cerr << ']';
            if (!sub_type.empty())std::cerr << ' ' << sub_type << ':';
            std::cerr << ' ' << data << std::endl;
        }

        /**
         * @brief 打印一行错误日志
         * @details 2种输出形式
         * @details [type id] data
         * @details [type] data
         * @tparam T log内容类型
         * @param type log分类
         * @param id 分类ID(小于0不输出)
         * @param data log内容
         */
        template<typename T>
        inline void err(const std::string &type, int id, const T &data) { err(type, id, "", data); }

        /**
         * @brief 打印一行日志
         * @details 4种输出形式
         * @details [type id] sub_type data
         * @details [type] sub_type data
         * @details [type id] data
         * @details [type] data
         * @tparam T log内容类型
         * @param type log分类
         * @param sub_type log子分类(为空不输出)
         * @param data log内容
         * @param id 分类ID(小于0不输出)
         */
        template<typename T>
        inline void log_or_err(const std::string &type, const std::string &sub_type, const T &data) {
            if (data)log(type, sub_type, data);
            else err(type, sub_type, data);
        }

        /**
         * @brief 宏打印辅助函数
         * @details [file:line:func] expr -> result
         * @param file 文件名
         * @param line 行号
         * @param func 方法名
         * @param expr 表达式
         * @param data 表达式结果
         */
        template<typename T>
        inline void log_loc(const std::string &file, const int &line, const std::string &func, const std::string &expr,
                            const T &data) {
            std::lock_guard<std::mutex> lock(getLogMtx());
            print_time(std::cout);
            std::cout << '[' << getFile(file) << ':' << line << ':' << func << "] "
                      << expr << " -> " << data << std::endl;
        }

        /**
         * @brief 宏打印辅助函数
         * @details [file:line:func] expr -> result
         * @param file 文件名
         * @param line 行号
         * @param func 方法名
         * @param expr 表达式
         * @param data 表达式结果
         */
        template<typename T>
        inline void err_loc(const std::string &file, const int &line, const std::string &func, const std::string &expr,
                            const T &data) {
            std::unique_lock<std::mutex> lock(getLogMtx());
            print_time(std::cerr);
            std::cerr << '[' << getFile(file) << ':' << line << ':' << func << "] "
                      << expr << " -> " << data << std::endl;
        }
    }
} // ifr

#endif //COMMON_MODULES_LOGGER_H
