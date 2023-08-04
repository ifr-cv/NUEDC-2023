#ifndef IFR_ROS2_CV__PACKAGE_IFR_COMMON__EXT_FUNCS__H
#define IFR_ROS2_CV__PACKAGE_IFR_COMMON__EXT_FUNCS__H

//MICROSOFT IS A DISASTER

#if !defined(__KERNEL__)
#  if !defined(__cplusplus)
#    include <stdio.h>
#  else
#    include <cstdio>
#  endif
#endif

#if !defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__) \
   && !defined(_MSC_VER)
#  pragma message ("Unsupported compiler. Some extended grammars not applied. Supported compilers: GCC, MinGW, MSVC")
#endif


#if !defined(__x86_64__) && !defined(_M_AMD64) && !defined(_M_X64) && !defined(__ARM_ARCH) \
   && !defined(_M_ARM)
#  pragma message ("Unrecognized architecture. Some functions might go wrong. Supported architectures: x64, ARM")
#  pragma message ("If you are exactly using x64 or ARM, please contact with the developer.")
#endif

#if !defined(__COMPILER_NAME__)
#  if defined(__GNUC__)
#    define __COMPILER_NAME__ "GCC/G++"
#    define GNU_GRAMMAR 1
#  elif defined(_MSC_VER)
#    define __COMPILER_NAME__ "MSC"
#    define MS_GRAMMAR 1
#  elif defined(__clang__)
#    define __COMPILER_NAME__ "clang/clang++"
#    define CLANG_GRAMMAR 1
#  else
#    define __COMPILER_NAME__ "Unknown"
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro '__COMPILER_NAME__' has already be defined, which is used in ext_funcs.h")
#endif

#define __COMPILER_GCC__      0
#define __COMPILER_MSC__      1
#define __COMPILER_CLANG__    2
#define __COMPILER_Unknown__ -1

#if !defined(__COMPILER__)
#  if defined(__GNUC__)
#    define __COMPILER__ __COMPILER_GCC__
#    define GNU_GRAMMAR 1
#  elif defined(_MSC_VER)
#    define __COMPILER__ __COMPILER_MSC__
#    define MS_GRAMMAR 1
#  elif defined(__clang__)
#    define __COMPILER__ __COMPILER_CLANG__
#    define CLANG_GRAMMAR 1
#  else
#    define __COMPILER__ __COMPILER_Unknown__
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro '__COMPILER__' has already be defined, which is used in ext_funcs.h")
#endif

#if !defined(__OS_NAME__)
#  if defined(__APPLE__)
#    define __OS_NAME__ "MacOS"
#  elif defined(__ANDROID__)
#    define __OS_NAME__ "Android"
#  elif defined(_WIN64) || defined(_WIN32)
#    define __OS_NAME__ "Windows"
#  elif defined(__linux__)
#    define __OS_NAME__ "Linux"
#  elif defined(__sun)
#    define __OS_NAME__ "Solaris"
#  elif defined(__unix__)
#    define __OS_NAME__ "UNIX"
#  else
#    define __OS_NAME__ "Unknown"
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro '__OS_NAME__' has already be defined, which is used in ext_funcs.h")
#endif

#define __OS_MacOS__    0
#define __OS_Android__  1
#define __OS_Windows__  2
#define __OS_Linux__    3
#define __OS_Solaris__  4
#define __OS_UNIX__     5
#define __OS_Unknown__ -1

#if !defined(__OS__)
#  if defined(__APPLE__)
#    define __OS__ __OS_MacOS__
#  elif defined(__ANDROID__)
#    define __OS__ __OS_Android__
#  elif defined(_WIN64) || defined(_WIN32)
#    define __OS__ __OS_Windows__
#  elif defined(__linux__)
#    define __OS__ __OS_Linux__
#  elif defined(__sun)
#    define __OS__ __OS_Solaris__
#  elif defined(__unix__)
#    define __OS__ __OS_UNIX__
#  else
#    define __OS__ __OS_Unknown__
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro '__OS__' has already be defined, which is used in ext_funcs.h")
#endif

#if !defined(__ARCH_NAME__)
#  if defined(__x86_64__) || defined(_M_AMD64) || defined(_M_X64)
#    define __ARCH_NAME__ "x64"  ///< Actually x64 sometimes refers to both ia64 and amd64. Here we treat the same
#  elif defined(_M_ARM) || defined(__ARM_ARCH)
#    define __ARCH_NAME__ "ARM"
#  else
#    define __ARCH_NAME__ "Unknown"
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro '__ARCH_NAME__' has already be defined, which is used in ext_funcs.h")
#endif

#define __ARCH_x64__      0
#define __ARCH_ARM__      1
#define __ARCH_Unknown__ -1

#if !defined(__ARCH__)
#  if defined(__x86_64__) || defined(_M_AMD64) || defined(_M_X64)
#    define __ARCH__ __ARCH_x64__  ///< Actually x64 sometimes refers to both ia64 and amd64. Here we treat the same
#  elif defined(_M_ARM) || defined(__ARM_ARCH)
#    define __ARCH__ __ARCH_ARM__
#  else
#    define __ARCH__ __ARCH_Unknown__
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro '__ARCH__' has already be defined, which is used in ext_funcs.h")
#endif

#if defined(MS_GRAMMAR)
#  pragma warning(disable:4996) //Damn _CRT_SECURE_NO_WARNINGS
#  pragma warning(disable:4267) //type cast
#  pragma warning(disable:4819) //file includes characters not in code page 936
#  define NOMINMAX
#endif

#if !defined(__EXPORT_C_START)
#  if defined(__cplusplus)
#    define __EXPORT_C_START extern "C"{
#  else
#    define __EXPORT_C_START 
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro '__EXPORT_C_START' has already be defined, which is used in ext_funcs.h")
#endif

#if !defined(__EXPORT_C_END)
#  if defined(__cplusplus)
#    define __EXPORT_C_END }
#  else
#    define __EXPORT_C_END 
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro '__EXPORT_C_END' has already be defined, which is used in ext_funcs.h")
#endif

#if defined(__KERNEL__)
#  if !define(KERN_EMERG)
#    define KERN_EMERG 
//Well I guess your kernel should have gone down when a KERN_EMERG is needed huh?
#  endif
#  if !define(KERN_ALERT)
#    define KERN_ALERT 
#  endif
#  if !define(KERN_CRTI)
#    define KERN_CRIT 
#  endif
#  if !define(KERN_ERR)
#    define KERN_ERR 
#  endif
#  if !define(KERN_WARNING)
#    define KERN_WARNING 
#  endif
#  if !define(KERN_NOTICE)
#    define KERN_NOTICE 
#  endif
#  if !define(KERN_INFO)
#    define KERN_INFO 
#  endif
#  if !define(KERN_DEBUG)
#    define KERN_DEBUG 
#  endif
#endif

#if !defined(__GENERAL_PRINT)
#  if defined(__KERNEL__)
#    define __GENERAL_PRINT(...) printk(__VA_ARGS__)
#  else
#    define __GENERAL_PRINT(...) printf(__VA_ARGS__)
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro '__GENERAL_PRINT' has already be defined, which is used in ext_funcs.h")
#endif
//This can hardly help but I guess someone may need this?

#if !defined(PRINT)
#  if defined(__KERNEL__)
#    if !defined(DFL_KERN_PRINT_LEVEL)
#      define DFL_KERN_PRINT_LEVEL KERN_INFO
#    endif
#    define PRINT(...) printk(DFL_KERN_PRINT_LEVEL,__VA_ARGS__)
#  else
#    define PRINT(...) printf(__VA_ARGS__)
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'PRINT' has already be defined, which is used in ext_funcs.h")
#endif

#if !defined(S_INF_OCTET)
#  define S_INF_OCTET 0x3f
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'S_INF_OCTET' has already be defined, which is used in ext_funcs.h")
#endif
//'S' stands for signed. Useful for graph

#if !defined(U_INF_OCTET)
#  define U_INF_OCTET 0x7f
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'U_INF_OCTET' has already be defined, which is used in ext_funcs.h")
#endif
//'U' stands for unsigned. Useful for graph

#if !defined(IS_POW2)
#  define IS_POW2(__value__) (!!((__value__)&((__value__)-1)))
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'IS_POW2' has already be defined, which is used in ext_funcs.h")
#endif

#if defined(PERMISSIVE_STH2BOOL)
#  if defined(Wdefine_ext_funcs)
#    pragma message ("Macro 'PERMISSIVE_STH2BOOL' has already be defined, which is used in ext_funcs.h")
#  endif
#else
#  if defined(STH2BOOL)
#    if defined(Wdefine_ext_funcs)
#      pragma message ("Macro 'STH2BOOL' has already be defined, which is used in ext_funcs.h")
#    endif
#  else
#    define PERMISSIVE_STH2BOOL(__value__) (!!__value__)
#    define STH2BOOL(__value__) PERMISSIVE_STH2BOOL(__value__)
#  endif
#endif

#if defined(PERMISSIVE_CAT)
#  if defined(Wdefine_ext_funcs)
#    pragma message ("Macro 'PERMISSIVE_CAT' has already be defined, which is used in ext_funcs.h")
#  endif
#else
#  if defined(STRCAT)
#    if defined(Wdefine_ext_funcs)
#      pragma message ("Macro 'STRCAT' has already be defined, which is used in ext_funcs.h")
#    endif
#  else
#    define PERMISSIVE_CAT(__Str1__, __Str2__) __Str1__##__Str2__
#    define STRCAT(__Str1__, __Str2__) PERMISSIVE_CAT(__Str1__, __Str2__)
#  endif
#endif

#if !defined(TEMP_NAME)
#  if defined(__COUNTER__)
#    define TEMP_NAME(__Name__) CATSTR(__Name__, __COUNTER__)
#  else
#    define TEMP_NAME(__Name__) CATSTR(__Name__, __LINE__)
#    if defined(Wdefine_ext_funcs)
#      pragma message ("Macro '__COUNTER__' was not defined, used '__LINE__' instead. This may cause problems.")
#    endif
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'TEMP_NAME' has already be defined, which is used in ext_funcs.h")
#endif

#if !defined(MEMBER_OFFSET)
#  define MEMBER_OFFSET(__Class__, __Member__) ((void*)(&((__Class__*)(NULL))->__Member__))
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'MEMBER_OFFSET' has already be defined, which is used in ext_funcs.h")
#endif

#if !defined(BUILD_BUG_ON_ZERO)
#  define BUILD_BUG_ON_ZERO(e) (sizeof(struct { int:-!!(e); }))
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'BUILD_BUG_ON_ZERO' has already be defined, which is used in ext_funcs.h")
#endif

#if !defined(BUILD_BUG_ON_NULL)
#  define BUILD_BUG_ON_NULL(e) ((void *)sizeof(struct { int:-!!(e); }))
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'BUILD_BUG_ON_NULL' has already be defined, which is used in ext_funcs.h")
#endif
//bitfield with width -1 causes an error, while e!=0

#if defined(GNU_GRAMMAR)
#  if !defined(LIKELY)
#    define LIKELY(__Cond__...) __builtin_expect(!!(__Cond__), 1)
#  elif defined(Wdefine_ext_funcs)
#    pragma message ("Macro 'LIKELY' has already be defined, which is used in ext_funcs.h")
#  endif
#  if !defined(UNLIKELY)
#    define UNLIKELY(__Cond__...) __builtin_expect(!!(__Cond__), 0)
#  elif defined(Wdefine_ext_funcs)
#    pragma message ("Macro 'UNLIKELY' has already be defined, which is used in ext_funcs.h")
#  endif
#else
#  if !defined(LIKELY)
#    define LIKELY(...) (__VA_ARGS__)
#  endif
#  if !defined(UNLIKELY)
#    define UNLIKELY(...) (__VA_ARGS__)
#  endif
#endif
//There's no such grammar in some compilers, thus there's no need to warn.

#if !defined(PREPOSE)
#  define PREPOSE(...) LIKELY(__VA_ARGS__)
#elif defined(GNU_GRAMMAR) && defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'PREPOSE' has already be defined, which is used in ext_funcs.h")
#endif
//!!(2)==1;  !!(1)==1;  !!(0)==0;

#if defined(DATA_SELECTOR)
#  if defined(Wdefine_ext_funcs)
#    pragma message ("Macro 'DATA_SELECTOR' has already be defined, which is used in ext_funcs.h")
#  endif
#else
#  if defined(__cplusplus)
#    define DATA_SELECTOR(__Var1__, __Var2__, ...) \
       ((__Cond__)?__Var1__:__VA_ARGS__)
#  else
#    if defined(GNU_GRAMMAR)
#      if !defined(__DATA_SELECTOR_PERMISSIVE)
#        define __DATA_SELECTOR_PERMISSIVE(__Var1__, __Var2__, __Temp__, __Cond__...) \
           *({ typeof(__Var1__)* __Temp__;\
             if(__Cond__)__Temp__=&__Var1__; \
             else __Temp__=&__Var2__; \
             __Temp__; \
           })
#        define DATA_SELECTOR(__Var1__, __Var2__, __Cond__...) \
           __DATA_SELECTOR_PERMISSIVE(__Var1__, __Var2__, TEMP_NAME(temp), __Cond__)
#      elif defined(Wdefine_ext_funcs)
#        pragma message ("Macro '__DATA_SELECTOR_PERMISSIVE' has already be defined, which is used in ext_funcs.h, \
           thus macro 'DATA_SELECTOR' was left unddefined")
#      endif
#    elif defined(MS_GRAMMAR)
#      define DATA_SELECTOR(__Var1__, __Var2__, ...) \
         ((__VA_ARGS__)?__Var1__:__Var2__)
#    endif
#  endif
#endif
// OK it seems that MSC does not differ C from C++?

#if defined(GNU_GRAMMAR)
#  define typeof(__Expression__...) typeof(__Expression__)
#elif defined(MS_GRAMMAR)
#  if !defined(PERMISSIVE_TYPEOF)
#    define PERMISSIVE_TYPEOF(...) decltype(__VA_ARGS__)
#    define typeof(...) PERMISSIVE_TYPEOF(__VA_ARGS__)
#  elif defined(Wdefine_ext_funcs)
#    pragma message ("Macro 'PERMISSIVE_TYPEOF' has already be defined, which is used in ext_funcs.h")
#  endif
#endif
//DO NOT use /Tc for MSV
//No need to warn about 'typeof'. If one want to redefine 'typeof', let them do it
//It's not about the English grammar stuff. You know 'singular they' used when one's gender is indetermined?

#if !defined(NAMED_STRUCT)
#  define NAMED_STRUCT(__Name__, ...) \
     typedef __VA_ARGS__ __Name__
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'NAMED_STRUCT' has already be defined, which is used in ext_funcs.h")
#endif
//__VA_ARGS__ refers to your definition.
//Compilers provided by Microsoft only provide with limited preprocessors.
//Their(Microsoft's) preprocessors are really f**king awful, seriously.
//When I tried to use a macro identifying the amount of arguments it receives,
//it worked well in GCC, but MSVC messed everything up. Yeah everything in __VA_ARGS__.
//I bet a senior of mine can write a better one.
//Well I admit that I can't. But who cares? Do I need to learn to keep food fresh if I've
//had enough of a f**king damn bad-designed fridge?
//MSVC really should have its preprocessor rewrote. How can you bear a preprocessor who 
//does not even support standard C & C++?

#if !defined(PACKED_STRUCT)
#  if defined(GNU_GRAMMAR)
#    define PACKED_STRUCT(__Declaration__...) \
       __Declaration__ \
       __attribute__((packed))
#  elif defined(MS_GRAMMAR)
#    define PACKED_STRUCT(...) \
       __pragma(pack(push,1)) \
       __VA_ARGS__ \
       __pragma(pack(pop))
#  else
#    define PACKED_STRUCT(...) __VA_ARGS__
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'PACKED_STRUCT' has already be defined, which is used in ext_funcs.h")
#endif

#if !defined(PACKED_NAMED_STRUCT)
#  if defined(GNU_GRAMMAR)
#    define PACKED_NAMED_STRUCT(__Name__, __Declaration__...) \
       typedef __Declaration__ \
       __attribute__((packed)) __Name__
#  elif defined(MS_GRAMMAR)
#    define PACKED_NAMED_STRUCT(__Name__, ...) \
       __pragma(pack(push,1)) \
       typedef __VA_ARGS__ __Name__\
       __pragma(pack(pop))
#  else
#    define PACKED_NAMED_STRUCT(__Name__, ...) \
       typedef __VA_ARGS__ __Name__
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'PACKED_NAMED_STRUCT' has already be defined, which is used in ext_funcs.h")
#endif

#if !defined(ALIGNED_STRUCT)
#  if defined(GNU_GRAMMAR)
#    define ALIGNED_STRUCT(__Length__, __Declaration__...) \
       __Declaration__ \
       __attribute__((aligned(__Length__),packed))
#  elif defined(MS_GRAMMAR)
#    define ALIGNED_STRUCT(__Length__, ...) \
       __pragma(pack(push,1)) \
       __declspec(align(__Length__)) \
       __VA_ARGS__ \
       __pragma(pack(pop))
#  else
#    define ALIGNED_STRUCT(__Length__, ...) __VA_ARGS__
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'ALIGNED_STRUCT' has already be defined, which is used in ext_funcs.h")
#endif

#if !defined(ALIGNED_NAMED_STRUCT)
#  if defined(GNU_GRAMMAR)
#    define ALIGNED_NAMED_STRUCT(__Length__, __Name__, __Declaration__...) \
       typedef __Declaration__ \
       __attribute__((aligned(__Length__),packed)) __Name__
#  elif defined(MS_GRAMMAR)
#    define ALIGNED_NAMED_STRUCT(__Length__, __Name__, ...) \
       __pragma(pack(push,1)) \
       typedef __declspec(align(__Length__)) __VA_ARGS__ __Name__ \
       __pragma(pack(pop))
#  else
#    define ALIGNED_NAMED_STRUCT(__Length__, __Name__, __Declaration__...) \
       typedef __Declaration__ __Name__
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'ALIGNED_NAMED_STRUCT' has already be defined, which is used in ext_funcs.h")
#endif

#if !defined(NO_RETURN)
#  if defined(GNU_GRAMMAR)
#    define NO_RETURN __attribute__((noreturn))
#  elif defined(MS_GRAMMAR)
#    define NO_RETURN __declspec(noreturn)
#  else
#    define NO_RETURN 
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'NO_RETURN' has already be defined, which is used in ext_funcs.h")
#endif
//This macro should be preposed

#if !defined(NO_INLINE)
#  if defined(GNU_GRAMMAR)
#    define NO_INLINE __attribute__((noinline))
#  elif defined(MS_GRAMMAR)
#    define NO_INLINE __declspec(noinline)
#  else
#    define NO_INLINE 
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'NO_INLINE' has already be defined, which is used in ext_funcs.h")
#endif
//This macro should be preposed

#if !defined(FORCE_INLINE)
#  if defined(GNU_GRAMMAR)
#    define FORCE_INLINE __inline__ __attribute__((always_inline)) static
#  elif defined(MS_GRAMMAR)
#    define FORCE_INLINE __forceinline static
#  else
#    define FORCE_INLINE 
#  endif
#elif defined(Wdefine_ext_funcs)
#  pragma message ("Macro 'FORCE_INLINE' has already be defined, which is used in ext_funcs.h")
#endif
//Note that all FORCE_INLINE functions are static

//__KERNEL__ is used to specify if you are compiling a kernel module

#endif// IFR_ROS2_CV__PACKAGE_IFR_COMMON__EXT_FUNCS__H

