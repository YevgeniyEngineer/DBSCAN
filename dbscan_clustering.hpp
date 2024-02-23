#ifndef DBSCAN_CLUSTERING_HPP
#define DBSCAN_CLUSTERING_HPP

#include <nanoflann.hpp> // nanoflann::KDTreeEigenMatrixAdaptor

#include "circular_queue.hpp"     // CircularQueue
#include "dbscan_point_cloud.hpp" // PointCloud

#include <cstdint>  // std::int32_t, std::size_t
#include <iostream> // std::cout
#include <stdexcept>
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
          kdtree_index_(number_of_dimensions, points_, {MAX_LEAF_SIZE}),
          search_parameters_{IGNORE_CHECKS, USE_APPROXIMATE_SEARCH, SORT_RESULTS}, index_queue_{points_.size()}
    {
        if (min_cluster_size_ < 1)
        {
            throw std::runtime_error("Minimum cluster size should not be less than 1");
        }
        if (max_cluster_size_ < min_cluster_size_)
        {
            throw std::runtime_error("Maximum cluster size should not be less than minimum cluster size");
        }
        if (points_.empty())
        {
            return;
        }

        // Reserve some memory for processing
        search_results_.reserve(points_.size());
        is_visited_.reserve(points_.size());
        cluster_labels_.reserve(points_.size());
        cluster_indices_.reserve(points_.size());
    }

    ~DBSCANClustering() = default;

    const auto getClusterIndices() const
    {
        return cluster_indices_;
    }

    void formClusters()
    {
        cluster_indices_.clear();

        if (points_.empty())
        {
            return; // Nothing to cluster
        }

        if (points_.size() < min_cluster_size_)
        {
            return; // Cannot form valid clusters
        }

        // Cluster label
        std::int32_t cluster_label = -1;

        // Set visited array
        is_visited_.assign(points_.size(), false);

        // Array to keep track of labels
        cluster_labels_.assign(points_.size(), labels::UNDEFINED);

        // Start clustering
        for (std::uint32_t i = 0; i < points_.size(); ++i)
        {
            if (cluster_labels_[i] != labels::UNDEFINED)
            {
                continue; // Point has already been added to a cluster
            }

            // Find core neighbours
            search_results_.clear();
            const auto number_of_core_neighbours = kdtree_index_.radiusSearch(
                &points_[i][0], distance_threshold_squared_, search_results_, search_parameters_);

            // Check if noise (below threshold number of cluster points)
            if (number_of_core_neighbours < min_cluster_size_)
            {
                cluster_labels_[i] = labels::NOISE;
                is_visited_[i] = true;
                continue;
            }

            // Move to next cluster
            ++cluster_label;

            // Assign current point to a new cluster
            cluster_labels_[i] = cluster_label;
            is_visited_[i] = true;

            // Transfer to index queue
            for (const auto &[index, distance] : search_results_)
            {
                if (index != i)
                {
                    index_queue_.push(index);
                }
            }

            // Find neighbours of seed points
            while (!index_queue_.empty())
            {
                // Remove last element
                const std::uint32_t seed_index = index_queue_.front();
                index_queue_.pop();

                // Check if border point - assign noise point to this cluster
                if (labels::NOISE == cluster_labels_[seed_index])
                {
                    cluster_labels_[seed_index] = cluster_label;
                    is_visited_[seed_index] = true;
                    continue;
                }

                // If point has not been classified before
                if (labels::UNDEFINED == cluster_labels_[seed_index])
                {
                    cluster_labels_[seed_index] = cluster_label;
                    is_visited_[seed_index] = true;

                    // Find seed neighbours
                    search_results_.clear();
                    const auto number_of_seed_neighbours = kdtree_index_.radiusSearch(
                        &points_[seed_index][0], distance_threshold_squared_, search_results_, search_parameters_);

                    // If the seed point is a core point
                    if (number_of_seed_neighbours >= min_cluster_size_)
                    {
                        // For each seed of the seed
                        for (const auto &[index, distance] : search_results_)
                        {
                            // If this seed has not been visited
                            if (!is_visited_[index])
                            {
                                index_queue_.push(index);
                                is_visited_[index] = true;
                            }
                        }
                    }
                }
            }
        }

        // Transfer labels
        for (std::uint32_t i = 0; i < points_.size(); ++i)
        {
            if (cluster_labels_[i] >= 0)
            {
                cluster_indices_[cluster_labels_[i]].push_back(i);
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

    // For processing
    std::vector<std::pair<std::uint32_t, float>> search_results_;
    std::vector<std::int32_t> cluster_labels_;
    std::vector<bool> is_visited_;
    CircularQueue<std::uint32_t> index_queue_;

    // Output
    ClusterT cluster_indices_;
};
} // namespace clustering

#endif // DBSCAN_CLUSTERING_HPP
