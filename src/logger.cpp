//
// Created by yuanlu on 2023/3/3.
//
#include "logger.h"
#include "ctime"

namespace ifr {
    namespace logger {
        static std::mutex log_mtx;
        static std::size_t prefix_length = 0;

        std::mutex &getLogMtx() { return log_mtx; }

        void print_time(std::ostream &outer) {
            time_t rawtime;
            time(&rawtime);
            auto timeinfo = localtime(&rawtime);
            outer << '[' << timeinfo->tm_hour << ':' << timeinfo->tm_min << ':' << timeinfo->tm_sec << "] ";
        }

        std::string getFile(const std::string &str) {
            return str.substr(str.length() > prefix_length ? prefix_length : 0);
        }

        void setMainPath(const std::string &main) {
            if (main.find('/') != std::string::npos)prefix_length = main.find_last_of('/') + 1;
            else if (main.find('\\') != std::string::npos)prefix_length = main.find_last_of('\\') + 1;
            else prefix_length = 0;
        }
    }
}