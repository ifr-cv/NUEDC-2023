//
// Created by yuanlu on 2023/8/4.
//

#ifndef NUEDC_2023_PKG_H
#define NUEDC_2023_PKG_H

#include "inttypes.h"

namespace ifr::pkg {
#pragma pack(1)

    struct Move {
        static constexpr const uint8_t HEAD = 0x5D;
        static constexpr const int MAX_R = std::numeric_limits<uint16_t>::max();
        static constexpr const int MID_R = MAX_R / 2;
        uint8_t head = HEAD;
        char m = 'm';
        uint16_t x;
        uint16_t y;
        char r_m = 'm';
        uint16_t r_x;
        uint16_t r_y;

        template<class T>
        void set(T fx, T fy) {
            int _x = std::max(std::min(int(fx) + MID_R, MAX_R), 0);
            int _y = std::max(std::min(int(fy) + MID_R, MAX_R), 0);
            x = r_x = _x;
            y = r_y = _y;
        }
    };

#pragma pack()
}
#endif //NUEDC_2023_PKG_H
