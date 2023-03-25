#include "dbscan_nanoflann_3d.hpp"
#include "point_struct.hpp"
#include <chrono>
#include <iostream>
#include <random>

int main()
{
    using namespace clustering;

    std::int32_t number_of_points = 100000; // 50000;
    std::int32_t number_of_dimensions = 3;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    PointCloud<double> points;
    points.pts.reserve(number_of_points);

    for (auto i = 0; i < number_of_points; ++i)
    {
        points.pts.emplace_back(dist(gen), dist(gen), dist(gen));
    }

    double eps = 0.05;
    std::int32_t min_pts = 4;
    double average_time = 0;
    std::int32_t number_of_iterations = 100;
    for (std::int32_t i = 0; i < number_of_iterations; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        DBSCAN<double> dbscan(eps, min_pts, points);
        dbscan.formClusters();

        auto stop_time = std::chrono::high_resolution_clock::now();
        auto elapsed = static_cast<double>((stop_time - start_time).count()) / 1e9;
        std::cout << "Elapsed time: " << elapsed << " seconds\n";
        average_time += elapsed;
    }

    average_time /= number_of_iterations;
    std::cout << "Average time per loop: " << average_time << " seconds\n";

    return EXIT_SUCCESS;
}