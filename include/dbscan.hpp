#ifndef DBSCAN_HPP
#define DBSCAN_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <execution>
#include <iostream>
#include <iterator>
#include <nanoflann.hpp>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clustering
{
using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 3>;

class DBSCAN
{
  public:
    explicit DBSCAN(double distance_threshold, std::int32_t min_neighbour_points, const PointCloud &points)
        : distance_threshold_(distance_threshold), distance_threshold_squared_(distance_threshold * distance_threshold),
          min_neighbour_points_(min_neighbour_points), points_(points), kdtree_(3, std::cref(points_), 10)
    {
    }

    const auto getClusterIndicies() const
    {
        return cluster_indices_;
    }

    void formClusters()
    {
        // Set all initial labels to -2
        std::vector<std::int32_t> labels(points_.rows(), -2);

        // Vector that holds status flags specifying if points were removed from queue
        // std::vector<bool> removed(points_.rows(), false);

        // Reserve memory for neighbors
        std::vector<std::pair<Eigen::Index, double>> neighbors;
        neighbors.reserve(10000);

        std::vector<std::pair<Eigen::Index, double>> inner_neighbors;
        inner_neighbors.reserve(10000);

        nanoflann::SearchParams search_parameters(32, 0.0, true);

        double query_point[3];
        double inner_query_point[3];

        // Initial cluster counter
        std::int32_t label = 0;

        // Iterate over each point
        for (std::int32_t index = 0; index < points_.rows(); ++index)
        {

            // Check if label is not undefined
            if (labels[index] != -2)
            {
                continue;
            }

            // Get the query point for current index
            query_point[0] = points_(index, 0);
            query_point[1] = points_(index, 1);
            query_point[2] = points_(index, 2);

            // Find nearest neighbors within radius
            neighbors.clear();
            std::size_t number_of_neighbors =
                kdtree_.index->radiusSearch(query_point, distance_threshold_squared_, neighbors, search_parameters);

            // Check density
            if (number_of_neighbors < min_neighbour_points_)
            {
                // Label query point as noise
                labels[index] = -1;
                continue;
            }

            // Set the next cluster label
            ++label;

            // Label initial point
            labels[index] = label;

            // Exclude the first point from the radius search, and iterate over all neighbors
            for (const auto [neighbor_index, _] : neighbors)
            {
                // Change noise to border point
                auto &neighbor_label = labels[neighbor_index];
                if (neighbor_label == -1)
                {
                    neighbor_label = label;
                }

                // Previously processed, border point
                if (neighbor_label != -2)
                {
                    continue;
                }

                // Label neighbor
                neighbor_label = label;

                // Get the query point for current index
                inner_query_point[0] = points_(neighbor_index, 0);
                inner_query_point[1] = points_(neighbor_index, 1);
                inner_query_point[2] = points_(neighbor_index, 2);

                // Find neighbors
                inner_neighbors.clear();
                std::size_t number_of_inner_neighbors = kdtree_.index->radiusSearch(
                    inner_query_point, distance_threshold_squared_, inner_neighbors, search_parameters);

                // Density check, if inner_query_point is a core point
                if (number_of_inner_neighbors >= min_neighbour_points_)
                {
                    // Add new neighbors to the seed set
                    for (const auto [inner_neighbor_index, _] : inner_neighbors)
                    {
                        labels[inner_neighbor_index] = label;
                    }
                }
            }
        }

        // Iterate over indices and copy to cluster indices
        for (std::int32_t i = 0; i < labels.size(); ++i)
        {
            cluster_indices_[labels[i]].push_back(i);
        }

        std::cout << "Number of clusters: " << cluster_indices_.size() << std::endl;
    }

  private:
    double distance_threshold_;
    double distance_threshold_squared_;
    std::int32_t min_neighbour_points_;
    const PointCloud &points_;
    nanoflann::KDTreeEigenMatrixAdaptor<PointCloud, 3, nanoflann::metric_L2, true> kdtree_;
    std::unordered_map<std::int32_t, std::vector<std::int32_t>> cluster_indices_;
};
} // namespace clustering

#endif // DBSCAN_HPP