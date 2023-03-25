cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(dbscan)

# add_library(${PROJECT_NAME} MODULE)

add_executable(dbscan 
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/simple_example.cpp
)

target_include_directories(dbscan
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
