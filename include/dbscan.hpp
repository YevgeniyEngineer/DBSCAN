#ifndef DBSCAN_HPP
#define DBSCAN_HPP

#include "nanoflann/nanoflann.hpp" // nanoflann::KDTreeEigenMatrixAdaptor
#include "point_struct.hpp"        // PointCloud
#include <cstdint>                 // std::int32_t, std::size_t
#include <iostream>                // std::cout
#include <unordered_map>           // std::unordered_map
#include <utility>                 // std::pair
#include <vector>                  // std::vector

namespace clustering
{
namespace labels
{
constexpr std::int32_t UNDEFINED = -2;
constexpr std::int32_t NOISE = -1;
} // namespace labels

template <typename CoordinateType> class DBSCAN final
{
  public:
    constexpr static std::int32_t NUMBER_OF_DIMENSIONS = 3;
    constexpr static std::int32_t MAX_LEAF_SIZE = 10;
    constexpr static std::int32_t IGNORE_CHECKS = 32;
    constexpr static float USE_APPROXIMATE_SEARCH = 0.0f;
    constexpr static bool SORT_RESULTS = true;

    DBSCAN(const DBSCAN &) = delete;
    DBSCAN &operator=(const DBSCAN &) = delete;
    DBSCAN(DBSCAN &&) = delete;
    DBSCAN &operator=(DBSCAN &&) = delete;
    DBSCAN() = delete;

    explicit DBSCAN(const double distance_threshold, const std::int32_t min_neighbour_points,
                    const PointCloud<CoordinateType> &points)
        : distance_threshold_squared_(distance_threshold * distance_threshold),
          min_neighbour_points_(min_neighbour_points), points_(points),
          kdtree_index_(NUMBER_OF_DIMENSIONS /*dim*/, points_, {MAX_LEAF_SIZE /*max leaf*/}),
          search_parameters_{IGNORE_CHECKS /*disable checks*/, USE_APPROXIMATE_SEARCH /*use approximated search?*/,
                             SORT_RESULTS /*sort*/}
    {
    }

    ~DBSCAN() = default;

    const auto getClusterIndices() const
    {
        return cluster_indices_;
    }

    void formClusters()
    {
        // Must not have less that 2 points
        if (points_.pts.size() < 2)
        {
            return;
        }

        // Set all initial labels to UNDEFINED
        std::vector<std::int32_t> labels(points_.pts.size(), labels::UNDEFINED);
        auto labels_it = labels.begin();

        // Reserve memory for neighbors
        static std::vector<std::pair<std::uint32_t, CoordinateType>> neighbors;
        neighbors.reserve(1000);

        static std::vector<std::pair<std::uint32_t, CoordinateType>> inner_neighbors;
        inner_neighbors.reserve(1000);

        // Initial cluster counter
        std::int32_t label = 0;

        // Iterate over each point
        for (std::int32_t index = 0; index < points_.pts.size(); ++index)
        {
            // Check if label is not undefined
            auto current_labels_it = labels_it + index;
            if (*current_labels_it != labels::UNDEFINED)
            {
                continue;
            }

            // Find nearest neighbors within radius
            neighbors.clear();
            const std::size_t number_of_neighbors = kdtree_index_.radiusSearch(
                points_.pts[index].point, distance_threshold_squared_, neighbors, search_parameters_);

            // Check density
            if (number_of_neighbors < min_neighbour_points_)
            {
                // Label query point as noise
                *current_labels_it = labels::NOISE;
                continue;
            }

            // Set the next cluster label
            ++label;

            // Label initial point
            *current_labels_it = label;

            // Exclude the first point from the radius search, and iterate over all neighbors
            for (auto neighbor_it = neighbors.cbegin(); neighbor_it != neighbors.cend(); ++neighbor_it)
            {
                const auto &neighbor_index = (*neighbor_it).first;
                auto current_neighbor_labels_it = labels_it + neighbor_index;

                if (*current_neighbor_labels_it == labels::NOISE)
                {
                    // Change noise to border point
                    *current_neighbor_labels_it = label;
                    continue;
                }

                // Previously processed, border point
                if (*current_neighbor_labels_it != labels::UNDEFINED)
                {
                    continue;
                }

                // Label neighbor
                *current_neighbor_labels_it = label;

                // Find neighbors
                inner_neighbors.clear();
                const std::size_t number_of_inner_neighbors =
                    kdtree_index_.radiusSearch(points_.pts[neighbor_index].point, distance_threshold_squared_,
                                               inner_neighbors, search_parameters_);

                // Density check, if inner_query_point is a core point
                if (number_of_inner_neighbors >= min_neighbour_points_)
                {
                    // Add new neighbors to the seed set
                    for (auto inner_neighbor_it = inner_neighbors.cbegin() + 1;
                         inner_neighbor_it != inner_neighbors.cend(); ++inner_neighbor_it)
                    {
                        const auto &inner_neighbour_index = (*inner_neighbor_it).first;
                        *(labels_it + inner_neighbour_index) = label;
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
    const double distance_threshold_squared_;
    const std::int32_t min_neighbour_points_;
    const PointCloud<CoordinateType> &points_;
    const nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Adaptor<CoordinateType, PointCloud<CoordinateType>>,
                                              PointCloud<CoordinateType>, NUMBER_OF_DIMENSIONS /* dim */>
        kdtree_index_;
    const nanoflann::SearchParams search_parameters_;

    std::unordered_map<std::int32_t, std::vector<std::int32_t>> cluster_indices_;
};
} // namespace clustering

#endif // DBSCAN_HPP