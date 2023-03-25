#include "dbscan.hpp"       // DBSCAN
#include "point_struct.hpp" // PointCloud
#include <chrono>           // std::chrono
#include <iostream>         // std::cout
#include <random>           // std::random_device

using namespace clustering;
using PointCloudType = PointCloud<double>;

// Constants
namespace configuration_parameters
{
constexpr static int NUMBER_OF_POINTS = 100'000;
constexpr static int NUMBER_OF_ITERATIONS = 100;
} // namespace configuration_parameters

namespace dbscan_parameters
{
constexpr static double NEAREST_NEIGHBOR_PROXIMITY_SQUARED = 0.1;
constexpr static std::int32_t MINIMUM_POINTS_TO_FORM_CLUSTER = 3;
} // namespace dbscan_parameters

/// @brief Generate and return random 3D points
static inline void generateRandomData(PointCloudType &points)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    points.pts.clear();
    points.pts.reserve(configuration_parameters::NUMBER_OF_POINTS);
    for (auto i = 0; i < configuration_parameters::NUMBER_OF_POINTS; ++i)
    {
        const double x = dist(gen);
        const double y = dist(gen);
        const double z = dist(gen);
        points.pts.emplace_back(x, y, z);
    }
}

static inline double runAndTimeExecution(DBSCAN<double> &dbscan)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Call the method to form clusters
    dbscan.formClusters();

    auto stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed = static_cast<double>((stop_time - start_time).count()) / 1e9;
    std::cout << "Elapsed time: " << elapsed << " seconds\n";

    return elapsed;
}

int main(int argc, const char **const argv)
{
    // Generate point cloud data
    PointCloudType points;
    generateRandomData(points);

    // Run several iterations of DBSCAN
    double average_time = 0.0;
    for (std::int32_t i = 0; i < configuration_parameters::NUMBER_OF_ITERATIONS; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        DBSCAN<double> dbscan(dbscan_parameters::NEAREST_NEIGHBOR_PROXIMITY_SQUARED,
                              dbscan_parameters::MINIMUM_POINTS_TO_FORM_CLUSTER, points);

        average_time += runAndTimeExecution(dbscan);
    }

    average_time /= configuration_parameters::NUMBER_OF_ITERATIONS;
    std::cout << "Average time per loop: " << average_time << " seconds\n";

    return EXIT_SUCCESS;
}