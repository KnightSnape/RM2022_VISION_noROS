cmake_minimum_required(VERSION 3.16)
project(GMaster_CV_2022)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS
        -I$(GENICAM_ROOT_V2_3)/library/CPP/include
        -I$(DAHENG_ROOT)/sdk/include
        )
set(ngraph_DIR "/opt/intel/openvino_2022/runtime/cmake")
set(InferenceEngine_DIR "/opt/intel/openvino_2022/runtime/cmake")

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(Ceres REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
        #OpenVINO推理引擎的头文件
        /opt/intel/openvino_2022/runtime/include/ie/
        /opt/intel/openvino_2022/runtime/include/ngraph/
)
include_directories(
        ${OpenCV_LIBRARY_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CERES_INCLUDE_DIR}
        ${YAML_CPP_INCLUDE_DIR}
        camera/include
        armor/include
        others/
        kalman/include
        kalman/src
        comm/
        windmill/include
        particle
)

# OpenVINO
file(GLOB_RECURSE sourcefile
        "armor/src/*.cpp"
        "armor/include/*.h"
        "kalman/src/*.cpp"
        "kalman/include/*.h"
        "camera/src/*.cpp"
        "camera/include/*.h"
        "windmill/src/*.cpp"
        "particle/*.cpp"
        "comm/*.cpp"
        "comm/*.h"
        )
add_subdirectory(serial/lib)
add_executable(
        GMaster_CV_2022
        main.cpp
        armor/include/ArmorDetector.h
        kalman/include/Kalman.h
        kalman/include/Kalmanfilter.h
        kalman/include/EKF.h
        comm/Checksum.h
        comm/CommPort.h
        others/GlobalParams.h
        kalman/include/Trajectory.h
        windmill/include/windmill_detect.h
        windmill/include/windmill_run.h
        windmill/include/windmill_predict.h
        windmill/include/windmill_trajectory.h

        armor/src/ArmorDetector.cpp
        kalman/src/Kalmanfilter.cpp
        kalman/src/EKF.cpp
        comm/Checksum.cpp
        comm/CommPort.cpp
        kalman/src/Trajectory.cpp
        windmill/src/windmill_detect.cpp

        ${sourcefile}
        armor/src/ArmorYOLO.cpp armor/src/ArmorNewYOLO.cpp)

target_link_libraries(
        GMaster_CV_2022
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
        ${CERES_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        nlohmann_json nlohmann_json::nlohmann_json
        serial
        spdlog::spdlog
        -lglog
        -lgxiapi
        -lpthread
)