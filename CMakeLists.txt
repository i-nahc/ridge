cmake_minimum_required(VERSION 3.16)
project(ridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The model library. Pure C++, no CUDA. This is Phase 1.
add_library(ridge
  src/Model.cpp
  src/Hardware.cpp
  src/Mma.cpp)
target_include_directories(ridge PUBLIC include)

# The CLI.
add_executable(ridge-predict tools/ridge-predict.cpp)
target_link_libraries(ridge-predict PRIVATE ridge)

# Tests.
enable_testing()
add_executable(test_model test/test_model.cpp)
target_link_libraries(test_model PRIVATE ridge)
add_test(NAME test_model COMMAND test_model)

# bench/ is CUDA and is added in Phase 2, guarded by finding a CUDA toolkit.
# include(CheckLanguage)
# check_language(CUDA)
# if(CMAKE_CUDA_COMPILER)
#   enable_language(CUDA)
#   add_subdirectory(bench)
# endif()
