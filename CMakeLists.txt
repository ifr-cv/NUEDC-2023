cmake_minimum_required(VERSION 3.5)
#设置项目名称
set(PROJECT_NAME NUEDC_2023)
#建立项目
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_C_FLAGS -pthread)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)


project(${PROJECT_NAME} LANGUAGES CXX C CUDA)

set(CMAKE_SYSROOT "${SYSROOT_PATH}")
set(CMAKE_FIND_ROOT_PATH "${SYSROOT_PATH}" "${CMAKE_PREFIX_PATH}" "${TOOLCHAIN_PATH}")


if (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Werror=return-type -Wno-unused-variable")
    #    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror -Wall -Werror=return-type -Wno-unused-variable")
    if (CMAKE_BUILD_TYPE STREQUAL Release)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto -march=native")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -flto -march=native")
        add_definitions(-DDEBUG=false)
    elseif (CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto -march=native -g")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -flto -march=native -g")
        add_definitions(-DDEBUG=false)
    elseif (CMAKE_BUILD_TYPE STREQUAL MinSizeRel)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -flto -march=native")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os -flto -march=native")
        add_definitions(-DDEBUG=false)
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
        add_definitions(-DDEBUG=true)
    endif ()
else ()
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
    add_compile_definitions(DEBUG=true)
endif ()

#寻找CUDA库
find_package(CUDA REQUIRED)
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "     libraries: ${CUDA_LIBRARIES}")
message(STATUS "     include path: ${CUDA_INCLUDE_DIRS}")

if (CMAKE_BUILD_TYPE STREQUAL Release) # nvcc flags
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++${CMAKE_CXX_STANDARD})
elseif (CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -G;-g;-std=c++${CMAKE_CXX_STANDARD})
elseif (CMAKE_BUILD_TYPE STREQUAL MinSizeRel)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++${CMAKE_CXX_STANDARD})
else ()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fPIC;-G;-g;-std=c++${CMAKE_CXX_STANDARD})
endif ()

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

#寻找OpenCV库
FIND_PACKAGE(OpenCV REQUIRED)
MESSAGE(STATUS "OpenCV version: ${OpenCV_VERSION}")
MESSAGE(STATUS "       libraries: ${OpenCV_LIBRARY_DIRS}")
MESSAGE(STATUS "       include path: ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "       libs: ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})


#打印调试信息
MESSAGE(STATUS "Project: ${PROJECT_NAME}")


include_directories(lib/serial/include)
include_directories(inc)
if (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
    include_directories(lib/Daheng/arm-linux)
else ()
    include_directories(lib/Daheng/win)
endif ()


aux_source_directory(${PROJECT_SOURCE_DIR}/lib/basic LIB_BASIC)
aux_source_directory(${PROJECT_SOURCE_DIR}/lib/SORT/sort-c++ LIB_SORT)

add_subdirectory(${PROJECT_SOURCE_DIR}/lib/serial)

file(GLOB_RECURSE SRCX src/*.cpp src/*.cu)
add_executable(${PROJECT_NAME} ${SRCX})


set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70)

target_link_libraries(${PROJECT_NAME}
        serial
        ${OpenCV_LIBS}
        ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_LIBRARY}
        )

if (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} gxiapi.so -lpthread -lm)
    set_target_properties(${PROJECT_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            )
else ()
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
            "$ENV{Daheng_Imaging_GalaxySDK}\\Samples\\VC SDK\\lib\\x64\\GxIAPI.lib"
            "$ENV{Daheng_Imaging_GalaxySDK}\\Samples\\VC SDK\\lib\\x64\\DxImageProc.lib"
            )
    set_target_properties(${PROJECT_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            )
endif ()