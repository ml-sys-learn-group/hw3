cmake_minimum_required(VERSION 3.2)
project(needle C CXX)

# find correct version of Python
#execute_process(COMMAND python3-config --prefix
#  OUTPUT_VARIABLE Python_ROOT_DIR)
# 
if(UNIX)
  set(Python_ROOT_DIR /home/hzx/miniconda3/envs/needle-env)
  set(PYTHON_LIB_PATH ${Python_ROOT_DIR}/lib)
  set(PYTHON_INCLUDE_PATH ${Python_ROOT_DIR}/include/python3.8)
  set(PYBIND_INCLUDE_PATH ${Python_ROOT_DIR}/lib/python3.8/site-packages/pybind11/include)
elseif (WIN32)
  set(Python_ROOT_DIR C:/Users/toofo/anaconda3/envs/needle-env)
  set(PYTHON_LIB_PATH ${Python_ROOT_DIR}/)
  set(PYTHON_INCLUDE_PATH ${Python_ROOT_DIR}/include/)
  set(PYBIND_INCLUDE_PATH ${Python_ROOT_DIR}/Lib/site-packages/pybind11/include)
elseif (APPLE)
  # TODO
  set(Python_ROOT_DIR C:/Users/toofo/anaconda3/envs/needle-env)
endif()

message(STATUS ${Python_ROOT_DIR})

include_directories(${PYTHON_INCLUDE_PATH})


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

include_directories(SYSTEM ${PYBIND_INCLUDE_PATH})
list(APPEND LINKER_LIBS ${PYTHON_LIB_PATH})


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc src/ndarray_backend_cpu.h)
#target_link_libraries(ndarray_backend_cpu PUBLIC )


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
  PREFIX ""
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()


####################
### CUDA BACKEND ###
####################
# find_package(CUDA)
# if(CUDA_FOUND)
#   message(STATUS "Found cuda, building cuda backend")

#   include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
#   list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

#   # invoke nvidia smi to detect if we really have a GPU
#   execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
#   if(NV_RET EQUAL "0")
#     CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
#   else()
#     # set to 3.7 the flag of K80
#     CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
#   endif()

#   # set arch flags properly
#   CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

#   target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
#   # pybind11_extension(ndarray_backend_cuda)
#   # pybind11_strip(ndarray_backend_cuda)

#   # directly output to ffi folder
#   set_target_properties(ndarray_backend_cuda
#     PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
#     CXX_VISIBILITY_PRESET "hidden"
#     CUDA_VISIBILITY_PRESET "hidden"
# )

# endif()

####################
### gtest ###
####################
message(STATUS ${LINKER_LIBS})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest)
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/googletest/include
)
link_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/googletest
        ${PYTHON_LIB_PATH}
)

enable_testing()
add_executable(ndarray_backend_test src/ndarray_backend_cpu.cc tests/ndarray_backend_cpu_test.cpp src/ndarray_backend_cpu.h)
target_link_libraries(ndarray_backend_test gtest python3.8)

include(GoogleTest)
gtest_discover_tests(ndarray_backend_test)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)