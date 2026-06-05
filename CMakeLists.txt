cmake_minimum_required(VERSION 3.14)
project(PlatEMO_CPP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen)

# ── snn-simulator static library (only the files needed for integration) ──────
set(SNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../snn-simulator")

add_library(snn_libs STATIC
    ${SNN_ROOT}/src/core/neuron.cpp
    ${SNN_ROOT}/src/core/network.cpp
    ${SNN_ROOT}/src/encoding/encoder.cpp
    ${SNN_ROOT}/src/encoding/rateEncoder.cpp
    ${SNN_ROOT}/src/encoding/poissonEncoder.cpp
    ${SNN_ROOT}/src/encoding/ttfsEncoder.cpp
    ${SNN_ROOT}/src/decoding/classificationDecoder.cpp
    ${SNN_ROOT}/src/core/simulator.cpp
)
target_include_directories(snn_libs PUBLIC ${SNN_ROOT}/include)
target_compile_options(snn_libs PRIVATE -O2 -Wall)

find_package(OpenMP REQUIRED)

# ─────────────────────────────────────────────────────────────────────────────
add_executable(platemo_cpp src/main.cpp)
target_include_directories(platemo_cpp PRIVATE include ${SNN_ROOT}/include)
target_link_libraries(platemo_cpp PRIVATE Eigen3::Eigen snn_libs OpenMP::OpenMP_CXX)
target_compile_options(platemo_cpp PRIVATE -O2 -Wall)
