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
          min_neighbour_points_(min_neighbour_points), points_(points), kdtree_(3, points_, 10)
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
        auto labels_it = labels.begin();

        // Vector that holds status flags specifying if points were removed from queue
        std::vector<bool> labelled(points_.rows(), false);
        auto labelled_it = labelled.begin();

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
            auto current_labelled_it = labelled_it + index;
            if (*current_labelled_it)
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
            auto current_labels_it = labels_it + index;
            if (number_of_neighbors < min_neighbour_points_)
            {
                // Label query point as noise
                (*current_labels_it) = -1;
                (*current_labelled_it) = true;
                continue;
            }

            // Set the next cluster label
            ++label;

            // Label initial point
            (*current_labels_it) = label;
            (*current_labelled_it) = true;

            // Exclude the first point from the radius search, and iterate over all neighbors
            for (auto neighbor_it = neighbors.cbegin(); neighbor_it != neighbors.cend(); ++neighbor_it)
            {
                // Change noise to border point
                const auto &neighbor_index = (*neighbor_it).first;
                auto current_neighbor_labels_it = labels_it + neighbor_index;
                auto current_neighbor_labelled_it = labelled_it + neighbor_index;

                if ((*current_neighbor_labels_it) < 0)
                {
                    (*current_neighbor_labels_it) = label;
                    (*current_neighbor_labelled_it) = true;
                    continue;
                }

                // Previously processed, border point
                if (*current_neighbor_labelled_it)
                {
                    continue;
                }

                // Label neighbor
                (*current_neighbor_labels_it) = label;
                (*current_neighbor_labelled_it) = true;

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
                    for (auto inner_neighbor_it = inner_neighbors.cbegin() + 1;
                         inner_neighbor_it != inner_neighbors.cend(); ++inner_neighbor_it)
                    {
                        const auto &inner_neighbour_index = (*inner_neighbor_it).first;
                        *(labels_it + inner_neighbour_index) = label;
                        *(labelled_it + inner_neighbour_index) = true;
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