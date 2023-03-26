#include "dbscan.hpp"       // DBSCAN
#include "point_struct.hpp" // PointCloud
#include <chrono>           // std::chrono
#include <csignal>          // std::signal
#include <iostream>         // std::cout
#include <random>           // std::random_device

using namespace clustering;

// Constants
namespace configuration_parameters
{
constexpr static int NUMBER_OF_DIMENSIONS = 3;
constexpr static int NUMBER_OF_POINTS = 100'000;
constexpr static int NUMBER_OF_ITERATIONS = 100;
} // namespace configuration_parameters

namespace dbscan_parameters
{
constexpr static double NEAREST_NEIGHBOR_PROXIMITY = 1.0;
constexpr static std::int32_t MINIMUM_POINTS_TO_FORM_CLUSTER = 10;
} // namespace dbscan_parameters

using CoordinateType = double;
using PointCloudType = PointCloud<CoordinateType, configuration_parameters::NUMBER_OF_DIMENSIONS>;

/// @brief Generate and return random 3D points
static inline void generateRandomData(PointCloudType &point_cloud)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<CoordinateType> dist(-10.0, 10.0);

    point_cloud.points.clear();
    point_cloud.points.reserve(configuration_parameters::NUMBER_OF_POINTS);
    for (auto i = 0; i < configuration_parameters::NUMBER_OF_POINTS; ++i)
    {
        std::array<CoordinateType, configuration_parameters::NUMBER_OF_DIMENSIONS> point_cache;
        for (auto j = 0; j < configuration_parameters::NUMBER_OF_DIMENSIONS; ++j)
        {
            point_cache[j] = dist(gen);
        }
        point_cloud.points.emplace_back(point_cache);
    }
}

static inline double runAndTimeExecution(DBSCAN<CoordinateType, configuration_parameters::NUMBER_OF_DIMENSIONS> &dbscan)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Call the method to form clusters
    dbscan.formClusters();

    auto stop_time = std::chrono::high_resolution_clock::now();
    auto elapsed = static_cast<double>((stop_time - start_time).count()) / 1e9;
    std::cout << "Elapsed time: " << elapsed << " seconds\n";

    return elapsed;
}

namespace
{
volatile std::sig_atomic_t signal_status;
}

int main(int argc, const char **const argv)
{
    // Install signal handler
    std::signal(SIGINT, [](int signal) {
        signal_status = signal;
        std::cout << "Keyboard interrupt\n";
        std::exit(signal);
    });

    // Generate point cloud data
    PointCloudType point_cloud;
    generateRandomData(point_cloud);

    // Run several iterations of DBSCAN
    double average_time = 0.0;
    for (std::int32_t i = 0; i < configuration_parameters::NUMBER_OF_ITERATIONS; ++i)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        DBSCAN<CoordinateType, configuration_parameters::NUMBER_OF_DIMENSIONS> dbscan(
            dbscan_parameters::NEAREST_NEIGHBOR_PROXIMITY, dbscan_parameters::MINIMUM_POINTS_TO_FORM_CLUSTER,
            point_cloud);

        average_time += runAndTimeExecution(dbscan);
    }

    average_time /= configuration_parameters::NUMBER_OF_ITERATIONS;
    std::cout << "Average time per loop: " << average_time << " seconds\n";

    return EXIT_SUCCESS;
}