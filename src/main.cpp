#include "dbscan.hpp"
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

    PointCloud points(number_of_points, number_of_dimensions);
    for (auto i = 0; i < number_of_points; ++i)
    {
        double x = dist(gen);
        double y = dist(gen);
        double z = dist(gen);
        Eigen::Vector3d point(x, y, z);
        points.row(i) = point;
    }

    double eps = 0.1;
    std::int32_t min_pts = 3;

    auto start_time = std::chrono::high_resolution_clock::now();

    DBSCAN dbscan(eps, min_pts, points);
    dbscan.formClusters();

    // dbscan.run();

    auto stop_time = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << (stop_time - start_time).count() / 1e9 << " seconds" << std::endl;

    // std::cout << "Number of cluster: " << dbscan.number_of_clusters() << std::endl;

    // for (const int label : dbscan.cluster_labels())
    // {
    //     cout << label << endl;
    // }

    return EXIT_SUCCESS;
}