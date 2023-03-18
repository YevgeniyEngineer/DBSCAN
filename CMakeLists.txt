cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(dbscan)

# add_library(${PROJECT_NAME} MODULE)

find_package(TBB REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nanoflann REQUIRED)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
    TBB::tbb
    Eigen3::Eigen
    nanoflann::nanoflann
)