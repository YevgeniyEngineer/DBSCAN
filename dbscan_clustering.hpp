#ifndef DBSCAN_CLUSTERING_HPP
#define DBSCAN_CLUSTERING_HPP

#include <nanoflann.hpp> // nanoflann::KDTreeEigenMatrixAdaptor

#include "dbscan_point_cloud.hpp" // PointCloud

#include <cstdint>       // std::int32_t, std::size_t
#include <iostream>      // std::cout
#include <unordered_map> // std::unordered_map
#include <utility>       // std::pair
#include <vector>        // std::vector

namespace clustering
{
namespace labels
{
static constexpr std::int32_t UNDEFINED = -2;
static constexpr std::int32_t NOISE = -1;
} // namespace labels

template <typename CoordinateType, std::size_t number_of_dimensions> class DBSCANClustering final
{
    using PointCloudT = DBSCANPointCloud<CoordinateType, number_of_dimensions>;
    using KdTreeT = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<CoordinateType, PointCloudT>,
                                                        PointCloudT, number_of_dimensions>;
    using IndicesT = std::vector<std::uint32_t>;
    using ClusterT = std::unordered_map<std::int32_t, IndicesT>;

  public:
    constexpr static std::int32_t MAX_LEAF_SIZE = 10;
    constexpr static std::int32_t IGNORE_CHECKS = 32;
    constexpr static float USE_APPROXIMATE_SEARCH = 0.0f;
    constexpr static bool SORT_RESULTS = false;

    DBSCANClustering(const DBSCANClustering &) = delete;
    DBSCANClustering &operator=(const DBSCANClustering &) = delete;
    DBSCANClustering(DBSCANClustering &&) = delete;
    DBSCANClustering &operator=(DBSCANClustering &&) = delete;
    DBSCANClustering() = delete;

    explicit DBSCANClustering(const PointCloudT &points, CoordinateType distance_threshold = 0,
                              std::uint32_t min_cluster_size = 1,
                              std::uint32_t max_cluster_size = std::numeric_limits<std::uint32_t>::max())

        : distance_threshold_squared_(distance_threshold * distance_threshold), min_cluster_size_(min_cluster_size),
          max_cluster_size_(max_cluster_size), points_(points),
          kdtree_index_(number_of_dimensions, points_, {MAX_LEAF_SIZE}), search_parameters_{IGNORE_CHECKS,
                                                                                            USE_APPROXIMATE_SEARCH,
                                                                                            SORT_RESULTS}
    {
    }

    ~DBSCANClustering() = default;

    const auto getClusterIndices() const
    {
        return cluster_indices_;
    }

    void formClusters()
    {
        cluster_indices_.clear();

        // Must not have less that 2 points
        const auto number_of_points = points_.size();
        if (number_of_points < 2 || min_cluster_size_ < 1)
        {
            return;
        }

        // Set all initial labels to UNDEFINED
        std::vector<std::int32_t> labels(number_of_points, labels::UNDEFINED);
        auto labels_it = labels.begin();

        // Reserve memory for neighbours
        std::vector<std::pair<std::uint32_t, CoordinateType>> neighbours;
        neighbours.reserve(number_of_points);

        std::vector<std::pair<std::uint32_t, CoordinateType>> inner_neighbours;
        inner_neighbours.reserve(number_of_points);

        // Initial cluster counter
        std::int32_t label = 0;

        // Iterate over each point
        for (std::uint32_t index = 0; index < number_of_points; ++index)
        {
            // Check if label is not undefined
            auto current_labels_it = labels_it + index;
            if (*current_labels_it != labels::UNDEFINED)
            {
                continue;
            }

            // Find nearest neighbours within radius
            neighbours.clear();

            // Check density
            const auto number_of_neighbours = kdtree_index_.radiusSearch(
                &points_[index][0], distance_threshold_squared_, neighbours, search_parameters_);

            // Check if a noise point
            if (number_of_neighbours < min_cluster_size_)
            {
                *current_labels_it = labels::NOISE;
                continue;
            }

            // Set the next cluster label
            ++label;

            // Label initial point
            *current_labels_it = label;

            // Exclude the first point from the radius search, and iterate over all neighbours
            for (auto neighbour_it = neighbours.cbegin(); neighbour_it != neighbours.cend(); ++neighbour_it)
            {
                auto current_neighbor_labels_it = labels_it + neighbour_it->first;

                // Check if a noise point
                if (*current_neighbor_labels_it == labels::NOISE)
                {
                    *current_neighbor_labels_it = label;
                    continue;
                }

                // Check if previously processed
                if (*current_neighbor_labels_it != labels::UNDEFINED)
                {
                    continue;
                }

                // Label neighbor
                *current_neighbor_labels_it = label;

                // Find inner neighbours
                inner_neighbours.clear();

                const auto number_of_inner_neighbours =
                    kdtree_index_.radiusSearch(&points_[neighbour_it->first][0], distance_threshold_squared_,
                                               inner_neighbours, search_parameters_);

                // Density check, if inner_query_point is a core point
                if (number_of_inner_neighbours >= min_cluster_size_)
                {
                    // Add new neighbours to the seed set
                    for (auto inner_neighbour_it = inner_neighbours.cbegin();
                         inner_neighbour_it != inner_neighbours.cend(); ++inner_neighbour_it)
                    {
                        *(labels_it + inner_neighbour_it->first) = label;
                    }
                }
            }
        }

        // Iterate over indices and copy to cluster indices
        if (!labels.empty())
        {
            ClusterT temp_cluster_indices;
            temp_cluster_indices.reserve(labels.back() + 3);
            for (std::uint32_t i = 0; i < labels.size(); ++i)
            {
                temp_cluster_indices[labels[i]].push_back(i);
            }

            cluster_indices_.reserve(temp_cluster_indices.size());
            for (auto &[label, indices] : temp_cluster_indices)
            {
                if (indices.size() > max_cluster_size_)
                {
                    continue;
                }

                cluster_indices_[label] = std::move(indices);
            }
        }
    }

  private:
    // Inputs
    CoordinateType distance_threshold_squared_;
    std::uint32_t min_cluster_size_;
    std::uint32_t max_cluster_size_;
    PointCloudT points_;
    KdTreeT kdtree_index_;
    nanoflann::SearchParams search_parameters_;

    // Output
    ClusterT cluster_indices_;
};
} // namespace clustering

#endif // DBSCAN_CLUSTERING_HPP