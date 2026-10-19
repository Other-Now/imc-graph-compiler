cmake_minimum_required(VERSION 3.20)
project(imc-graph-compiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "MLIR at ${MLIR_DIR}, LLVM ${LLVM_PACKAGE_VERSION}")
if(NOT LLVM_VERSION_MAJOR EQUAL 20)
  message(WARNING "developed against LLVM/MLIR 20, found ${LLVM_PACKAGE_VERSION}")
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/Imc)

# Link the single libMLIR.so / libLLVM.so instead of ~370 static archives:
# keeps the link inside a 2 GB WSL VM and takes seconds.
add_executable(imc-opt tools/imc-opt.cpp lib/ImcDialect.cpp lib/ImcPasses.cpp)
add_dependencies(imc-opt ImcIncGen ImcPassesIncGen)
target_link_libraries(imc-opt PRIVATE MLIR LLVM)
