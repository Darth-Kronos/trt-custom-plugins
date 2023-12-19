cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(CustomCosLUPlugin)

find_package(Torch REQUIRED)

# Define our library target
add_library(CustomCosLUPlugin SHARED op.cpp)

# Enable C++14
target_compile_features(CustomCosLUPlugin PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(CustomCosLUPlugin "${TORCH_LIBRARIES}")