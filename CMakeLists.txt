cmake_minimum_required(VERSION 3.18)
project(haversine_project LANGUAGES CXX CUDA)

# 1. Find Python (Required for Pybind11)
find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)

# 2. Find Pybind11
# We use python to find the cmake path to ensure we get the pip-installed version
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED CONFIG)

# 3. Create the Python Module
# IMPORTANT: pybind11_add_module automatically handles linking Python libraries
# and setting the correct file extensions (.so vs .pyd, etc.)
pybind11_add_module(haversine_library 
    haversine_library.cpp 
    haversine_library.cu
)

# 4. Set CUDA Properties
# CUDA_SEPARABLE_COMPILATION is often needed for complex kernels, 
# and POSITION_INDEPENDENT_CODE is mandatory for shared libraries.
set_target_properties(haversine_library PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 5. Include Directories
# Ensure the target can see CUDA headers
target_include_directories(haversine_library PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
