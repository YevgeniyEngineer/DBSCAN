#ifndef POINT_STRUCT_HPP
#define POINT_STRUCT_HPP

#include <cstdint>   // std::size_t
#include <stdexcept> // std::runtime_error
#include <vector>    // std::vector

namespace clustering
{
/// @brief Point Struct defined for 3 dimensions
template <typename CoordinateType> struct PointCloud
{
    /// @brief Definition of the point struct
    struct Point
    {
        Point() = delete;
        explicit Point(CoordinateType x, CoordinateType y, CoordinateType z) : point{x, y, z}
        {
        }
        CoordinateType point[3];
    };

    // Container for points
    std::vector<Point> points;

    /// @brief Return the number of points in the cloud
    inline std::size_t kdtree_get_point_count() const noexcept
    {
        return points.size();
    }

    /// @brief Get a point along the specified dimension
    inline CoordinateType kdtree_get_pt(const std::size_t idx, const std::size_t dim) const
    {
        switch (dim)
        {
        case (0UL): {
            return points[idx].point[0];
        }
        case (1UL): {
            return points[idx].point[1];
        }
        case (2UL): {
            return points[idx].point[2];
        }
        default: {
            throw std::runtime_error("Attempting to access the wrong dimension!");
        }
        }
    }

    /// @brief Optional bounding box computation
    template <class Bbox> inline bool kdtree_get_bbox(Bbox & /* bb */) const
    {
        return false;
    }
};
} // namespace clustering

#endif // POINT_STRUCT_HPP