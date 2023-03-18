#ifndef DBSCAN_HPP
#define DBSCAN_HPP

// STL
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

// Eigen
#include <eigen3/Eigen/Dense>

// Nanoflann
#include <nanoflann.hpp>

// TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace clustering
{
using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 3>;

class DBSCAN
{
  public:
    explicit DBSCAN(double eps, std::int32_t min_pts, const PointCloud &points)
        : eps_(eps), min_pts_(min_pts), points_(points),
          kdtree_(
              3 /*number of dimensions */, std::cref(points_),
              10 /* max leaf */) // See https://github.com/jlblancoc/nanoflann/blob/master/examples/matrix_example.cpp
    {
        cluster_labels_.resize(points_.rows(), -1);
    }

    void run()
    {
        std::int32_t cluster_id = 0;

        tbb::parallel_for(tbb::blocked_range<std::int32_t>(0, points_.rows()),
                          [&](const tbb::blocked_range<std::int32_t> &r) {
                              for (std::int32_t i = r.begin(); i != r.end(); ++i)
                              {
                                  if (cluster_labels_[i] != -1)
                                  {
                                      continue;
                                  }
                                  std::vector<std::pair<Eigen::Index, double>> neighbors = regionQuery(i);
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

    std::int32_t min_pts() const
    {
        return min_pts_;
    }

    std::vector<std::int32_t> cluster_labels() const
    {
        return cluster_labels_;
    }

  private:
    void expandCluster(int point_idx, std::vector<std::pair<Eigen::Index, double>> &neighbors, std::int32_t cluster_id)
    {
        cluster_labels_[point_idx] = cluster_id;

        for (std::size_t i = 0; i < neighbors.size(); ++i)
        {
            int n = static_cast<int>(neighbors[i].first);
            if (cluster_labels_[n] == -1)
            {
                cluster_labels_[n] = cluster_id;
                std::vector<std::pair<Eigen::Index, double>> n_neighbors = regionQuery(n);
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

    std::vector<std::pair<Eigen::Index, double>> regionQuery(int point_idx)
    {
        std::vector<std::pair<Eigen::Index, double>> neighbors;
        double query_pt[3] = {points_(point_idx, 0), points_(point_idx, 1), points_(point_idx, 2)};

        nanoflann::SearchParams params;
        const double search_radius = std::pow(eps_, 2);
        const std::size_t num_results = kdtree_.index->radiusSearch(query_pt, search_radius, neighbors, params);

        // printf("Num results: %d\n", static_cast<int>(num_results));

        neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
                                       [&point_idx](std::pair<Eigen::Index, double> neighbor) {
                                           return (static_cast<std::int32_t>(neighbor.first) == point_idx);
                                       }),
                        neighbors.end());
        return neighbors;
    }

    double eps_;
    std::int32_t min_pts_;
    PointCloud points_;
    std::vector<std::int32_t> cluster_labels_;
    std::mutex mutex_;
    nanoflann::KDTreeEigenMatrixAdaptor<PointCloud, 3 /* number of dimensions */, nanoflann::metric_L2,
                                        true /* row major layout */>
        kdtree_;
};

} // namespace clustering

#endif // DBSCAN_HPP