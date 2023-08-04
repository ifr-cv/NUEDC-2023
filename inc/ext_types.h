#ifndef IFR_ROS2_CV__PACKAGE_IFR_COMMON__EXT_TYPES__H
#define IFR_ROS2_CV__PACKAGE_IFR_COMMON__EXT_TYPES__H
#if !defined(__KERNEL__)
#include <stdint.h>
#endif
#include "ext_funcs.h"

#if defined(__cplusplus)
#include <stdbool.h>
#endif

#if __OS__ != __OS_Windows__
typedef uint8_t BYTE;
typedef uint16_t WORD;
typedef uint32_t DWORD;
#endif
//I guess nobody want to use C++ codes as C codes?
//OK man I ... Really someone does. OK. Whatever. Let it go.
#if __COMPILER__ == __COMPILER_MSC__ && defined(__cplusplus)
typedef bool _Bool;
#endif

#if __OS__ == __OS_Windows__
typedef void(__cdecl *sighandler_t)(int);
#endif

typedef _Bool bit;
typedef uint8_t octet_t;

#define SUCCESS_CODE 0
#define ERR_NO_SUCH_POS -1

//In case one day we have to remove the 'FORCE_INLINE' declaration we use 'EXPORT_C'
__EXPORT_C_START

FORCE_INLINE int modify_octet(octet_t *dst, uint8_t pos, bit value) {
    if (UNLIKELY(pos > 7 || pos < 0))
        return ERR_NO_SUCH_POS;
        //This makes code clearer, I think
    else if (value)
        *dst |= (1 << pos);
    else
        *dst &= ~(1 << pos);
    //else *dst = (((*dst) & (0xff - (1 << pos))) | (value << pos));
    return SUCCESS_CODE;
}

FORCE_INLINE int assign_octet(octet_t *dst, octet_t src) {
    *dst = src;
    return SUCCESS_CODE;
}

FORCE_INLINE int get_bit_octet(octet_t src, uint8_t pos, bit *value) {
    if (UNLIKELY(pos > 7 || pos < 0))
        return ERR_NO_SUCH_POS;
    else
        *value = (!!(src & (1 << pos)));
    return SUCCESS_CODE;
}

FORCE_INLINE int reverse_octet(octet_t *dst) {
    *dst = ((((*dst) & 0x01) << 7) | (((*dst) & 0x02) << 5) | (((*dst) & 0x04) << 3) |
            (((*dst) & 0x08) << 1) | (((*dst) & 0x10) >> 1) | (((*dst) & 0x20) >> 3) |
            (((*dst) & 0x40) >> 5) | (((*dst) & 0x80) >> 7));
    return SUCCESS_CODE;
}

int reverse_octets_bits(octet_t *dst, size_t size);

int reverse_octets(octet_t *dst, size_t size);

int put_octet(octet_t src);
//Note that this function does NOT flush the output buffer

int put_octets(octet_t *src, size_t size, char sep);

__EXPORT_C_END

#endif// IFR_ROS2_CV__PACKAGE_IFR_COMMON__EXT_TYPES__H