#ifndef DBSCAN_HPP
#define DBSCAN_HPP

// STL
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
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
                          [&](const tbb::blocked_range<std::int32_t> &index_range) {
                              for (std::int32_t i = index_range.begin(); i != index_range.end(); ++i)
                              {
                                  if (cluster_labels_[i] != -1)
                                  {
                                      continue;
                                  }
                                  std::vector<std::pair<Eigen::Index, double>> neighbors = regionQuery(i);
                                  if (neighbors.size() < min_pts_)
                                  {
                                      std::unique_lock<std::mutex> lock(mutex_);
                                      //   std::cout << "Here" << std::endl;
                                      cluster_labels_[i] = 0; // Noise
                                  }
                                  else
                                  {
                                      std::unique_lock<std::mutex> lock(mutex_);
                                      //   printf("%d\n", cluster_id);
                                      //   std::cout << "Here" << std::endl;
                                      expandCluster(i, neighbors, ++cluster_id);
                                  }
                              }
                          });

        number_of_clusters_ = cluster_id + 1;
    }

    double eps() const
    {
        return eps_;
    }

    std::int32_t min_pts() const
    {
        return min_pts_;
    }

    std::int32_t number_of_clusters() const
    {
        return number_of_clusters_;
    }

    std::vector<std::int32_t> cluster_labels() const
    {
        return cluster_labels_;
    }

  private:
    void expandCluster(std::int32_t point_idx, std::vector<std::pair<Eigen::Index, double>> &neighbors,
                       std::int32_t cluster_id)
    {
        cluster_labels_[point_idx] = cluster_id;

        for (std::size_t i = 0; i < neighbors.size(); ++i)
        {
            std::int32_t n = static_cast<std::int32_t>(neighbors[i].first);
            auto &cluster_label = cluster_labels_[n];
            if (cluster_label == -1)
            {
                cluster_label = cluster_id;
                std::vector<std::pair<Eigen::Index, double>> n_neighbors = regionQuery(n);
                if (n_neighbors.size() >= min_pts_)
                {
                    neighbors.insert(neighbors.end(), n_neighbors.begin(), n_neighbors.end());
                }
            }
            else if (cluster_label == 0)
            {
                cluster_label = cluster_id;
            }
        }
    }

    std::vector<std::pair<Eigen::Index, double>> regionQuery(std::int32_t point_idx)
    {
        std::vector<std::pair<Eigen::Index, double>> neighbors;
        neighbors.reserve(50);

        double query_pt[3] = {points_(point_idx, 0), points_(point_idx, 1), points_(point_idx, 2)};

        nanoflann::SearchParams params;
        params.eps = 2;
        params.sorted = true; // return sorted results - this will ensure that the first element index = point_idx

        const double squared_search_radius = std::pow(eps_, 2);
        const std::size_t num_results = kdtree_.index->radiusSearch(query_pt, squared_search_radius, neighbors, params);

        if (neighbors.size() <= 1)
        {
            return {};
        }
        else
        {
            std::vector<std::pair<Eigen::Index, double>> neighbors_without_first_index;
            neighbors_without_first_index.reserve(neighbors.size() - 1);
            std::copy(std::make_move_iterator(neighbors.begin() + 1), std::make_move_iterator(neighbors.end()),
                      std::back_inserter(neighbors_without_first_index));
            neighbors.clear();
            return neighbors_without_first_index;
        }
    }

    double eps_;
    std::int32_t min_pts_;
    std::int32_t number_of_clusters_;
    const PointCloud &points_;
    std::vector<std::int32_t> cluster_labels_;
    nanoflann::KDTreeEigenMatrixAdaptor<PointCloud, 3 /* number of dimensions */, nanoflann::metric_L2,
                                        true /* row major layout */>
        kdtree_;

    std::mutex mutex_;
};

} // namespace clustering

#endif // DBSCAN_HPP