cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(dbscan)

# add_library(${PROJECT_NAME} MODULE)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nanoflann REQUIRED)

add_executable(dbscan_nanoflann_3d 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(dbscan_nanoflann_3d
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dbscan_nanoflann_3d
    Eigen3::Eigen
    nanoflann::nanoflann
)