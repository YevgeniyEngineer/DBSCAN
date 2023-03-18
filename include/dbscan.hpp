#ifndef DBSCAN_HPP
#define DBSCAN_HPP

#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <vector>

namespace clustering
{
using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 3>;

class DBSCAN
{
  public:
    explicit DBSCAN(double eps, int min_pts, const PointCloud &points) : eps_(eps), min_pts_(min_pts), points_(points)
    {
        cluster_labels_.resize(points_.rows(), -1);
    }

    void run()
    {
        int cluster_id = 0;

        tbb::parallel_for(tbb::blocked_range<int>(0, points_.rows()), [&](const tbb::blocked_range<int> &r) {
            for (int i = r.begin(); i != r.end(); ++i)
            {
                if (cluster_labels_[i] != -1)
                {
                    continue;
                }
                std::vector<int> neighbors = regionQuery(i);
                if (neighbors.size() < min_pts_)
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cluster_labels_[i] = 0; // Noise
                }
                else
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    expandCluster(i, neighbors, ++cluster_id);
                }
            }
        });
    }

    double eps() const
    {
        return eps_;
    }

    int min_pts() const
    {
        return min_pts_;
    }

    std::vector<int> cluster_labels() const
    {
        return cluster_labels_;
    }

  private:
    void expandCluster(int point_idx, std::vector<int> &neighbors, int cluster_id)
    {
        cluster_labels_[point_idx] = cluster_id;

        for (size_t i = 0; i < neighbors.size(); ++i)
        {
            int n = neighbors[i];
            if (cluster_labels_[n] == -1)
            {
                cluster_labels_[n] = cluster_id;
                std::vector<int> n_neighbors = regionQuery(n);
                if (n_neighbors.size() >= min_pts_)
                {
                    neighbors.insert(neighbors.end(), n_neighbors.begin(), n_neighbors.end());
                }
            }
            else if (cluster_labels_[n] == 0)
            {
                cluster_labels_[n] = cluster_id;
            }
        }
    }

    std::vector<int> regionQuery(int point_idx)
    {
        std::vector<int> neighbors;
        for (int i = 0; i < points_.rows(); ++i)
        {
            if (i != point_idx && (points_.row(i) - points_.row(point_idx)).norm() <= eps_)
            {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    }

    double eps_;
    int min_pts_;
    PointCloud points_;
    std::vector<int> cluster_labels_;
    std::mutex mutex_;
};

} // namespace clustering

#endif // DBSCAN_HPP