cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cosLU)

find_package(Torch REQUIRED)

# Define our library target
add_library(cosLU SHARED op.cpp)

# Enable C++14
target_compile_features(cosLU PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(cosLU "${TORCH_LIBRARIES}")