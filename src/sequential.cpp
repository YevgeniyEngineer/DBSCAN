#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <nanoflann.hpp>
#include <random>
#include <utility>
#include <vector>

#define THREADED 0

typedef Eigen::Matrix<double, Eigen::Dynamic, 3> PointCloud;

struct PointCloudAdapter
{
    const PointCloud &points;

    PointCloudAdapter(const PointCloud &points) : points(points)
    {
    }

    inline size_t kdtree_get_point_count() const
    {
        return points.rows();
    }

    inline double kdtree_distance(const double *p1, const size_t idx_p2, size_t /*size*/) const
    {
        double d0 = p1[0] - points(idx_p2, 0);
        double d1 = p1[1] - points(idx_p2, 1);
        double d2 = p1[2] - points(idx_p2, 2);
        return d0 * d0 + d1 * d1 + d2 * d2;
    }

    inline double kdtree_get_pt(const size_t idx, int dim) const
    {
        return points(idx, dim);
    }

    template <class BBOX> bool kdtree_get_bbox(BBOX & /*bbox*/) const
    {
        return false;
    }
};

class DBSCAN
{
  public:
    constexpr static std::int32_t MAX_NEIGH_TO_SEARCH = 200;
    PointCloud points;
    double eps;
    double search_radius_sq;
    std::int32_t min_pts;
    std::int32_t neigh_to_search;
    std::int32_t number_of_clusters = 0;
    std::vector<std::int32_t> cluster_labels;
    PointCloudAdapter point_cloud_adapter;
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloudAdapter>, PointCloudAdapter, 3>
        kdtree;

    DBSCAN(double eps, std::int32_t min_pts, const PointCloud &points)
        : points(points), eps(eps), min_pts(min_pts), cluster_labels(points.rows(), 0), point_cloud_adapter(points),
          kdtree(3, point_cloud_adapter, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */))
    {
        kdtree.buildIndex();
        neigh_to_search = min_pts + 10;
        if (neigh_to_search > MAX_NEIGH_TO_SEARCH)
        {
            neigh_to_search = MAX_NEIGH_TO_SEARCH;
        }
        search_radius_sq = eps * eps;
    }

    void run()
    {
        std::int32_t cluster_id = 0;

        for (std::int32_t i = 0; i < points.rows(); ++i)
        {
            if (cluster_labels[i] == 0)
            {
                std::vector<std::int32_t> neighbors = regionQuery(i);
                if (neighbors.size() < min_pts)
                {
                    cluster_labels[i] = -1; // Mark as noise
                }
                else
                {
                    expandCluster(i, neighbors, ++cluster_id);
                }
            }
        }

        number_of_clusters = cluster_id + 1;
    }

    void expandCluster(std::int32_t core_point_index, std::vector<std::int32_t> &neighbors, std::int32_t cluster_id)
    {
        cluster_labels[core_point_index] = cluster_id;

        for (size_t i = 0; i < neighbors.size(); ++i)
        {
            std::int32_t neighbor_point_index = neighbors[i];
            if (cluster_labels[neighbor_point_index] == 0)
            {
                cluster_labels[neighbor_point_index] = cluster_id;
                std::vector<std::int32_t> neighbor_neighbors = regionQuery(neighbor_point_index);
                if (neighbor_neighbors.size() >= min_pts)
                {
                    neighbors.insert(neighbors.end(), neighbor_neighbors.begin(), neighbor_neighbors.end());
                }
            }
            else if (cluster_labels[neighbor_point_index] == -1)
            {
                cluster_labels[neighbor_point_index] = cluster_id;
            }
        }
    }

    std::vector<std::int32_t> regionQuery(std::int32_t point_idx)
    {
        static std::int32_t indices[MAX_NEIGH_TO_SEARCH];
        static double distances_sq[MAX_NEIGH_TO_SEARCH];

        const auto &point = points.row(point_idx);
        const double query_pt[3] = {point[0], point[1], point[2]};

        nanoflann::KNNResultSet<double, std::int32_t> result_set(neigh_to_search);
        result_set.init(indices, distances_sq);

        constexpr static bool sort_points = false;
        kdtree.findNeighbors(result_set, query_pt, nanoflann::SearchParams(32, 0.0f, sort_points));
        std::size_t number_of_neighbors = result_set.size();

        std::vector<std::int32_t> neighbors;
        neighbors.reserve(number_of_neighbors);
        for (std::int32_t i = 0; i < number_of_neighbors; ++i)
        {
            auto &index = indices[i];
            if (index == point_idx)
            {
                continue;
            }
            else if (distances_sq[i] < search_radius_sq)
            {
                neighbors.push_back(index);
            }
        }
        return neighbors;
    }
};

int main()
{
    double eps = 1.0;
    std::int32_t min_pts = 3;

    {
        // Create a sample point cloud
        PointCloud points(6, 3);
        points << 0, 0, 0, 1, 1, 1, 0, 1, 0, 8, 8, 8, 9, 9, 9, 8, 9, 8;

        DBSCAN dbscan(eps, min_pts, points);
        dbscan.run();

        // Print the cluster labels
        for (std::int32_t i = 0; i < dbscan.points.rows(); ++i)
        {
            std::cout << "Point " << i << ": [" << points(i, 0) << ", " << points(i, 1) << ", " << points(i, 2)
                      << "], Cluster: " << dbscan.cluster_labels[i] << std::endl;
        }
    }

    std::cout << "\n\n\n";

    {
        // Random points
        std::int32_t number_of_points = 100'000;
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

        std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();

        DBSCAN dbscan(eps, min_pts, points);
        dbscan.run();

        std::chrono::time_point<std::chrono::high_resolution_clock> t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Elapsed seconds for clustering: " << (t2 - t1).count() / 1e9 << std::endl;

        // Sort in ascending order of clusters
        std::vector<std::int32_t> indices(dbscan.points.rows());
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(indices.begin(), indices.end(), [&dbscan](std::int32_t i, std::int32_t j) {
            return dbscan.cluster_labels[i] < dbscan.cluster_labels[j];
        });

        // Sort values
        for (std::int32_t i = 0; i < dbscan.points.rows(); ++i)
        {
            std::int32_t j = indices[i];
            if (j != i)
            {
                auto row_i = dbscan.points.row(i);
                auto row_j = dbscan.points.row(j);
                dbscan.points.row(i) = row_j;
                dbscan.points.row(j) = row_i;
                std::swap(dbscan.cluster_labels[i], dbscan.cluster_labels[j]);
                std::swap(indices[i], indices[std::find(indices.begin() + i, indices.end(), i) - indices.begin()]);
            }
        }

        std::cout << "Number of clusters: " << dbscan.number_of_clusters << std::endl;

        // Print the cluster labels
        // for (std::int32_t i = 0; i < dbscan.points.rows(); ++i)
        // {
        //     std::cout << "Point " << i << ": [" << points(i, 0) << ", " << points(i, 1) << ", " << points(i, 2)
        //               << "], Cluster: " << dbscan.cluster_labels[i] << std::endl;
        // }
    }

    return 0;
}