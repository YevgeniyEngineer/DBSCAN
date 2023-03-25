#ifndef POINT_STRUCT_HPP
#define POINT_STRUCT_HPP

#include <cstdint> // std::size_t
#include <vector>  // std::vector

namespace clustering
{
/// @brief Point Struct defined for 3 dimensions
template <typename CoordinateType> struct PointCloud
{
    /// @brief Definition of the point struct
    struct Point
    {
        explicit Point(CoordinateType x, CoordinateType y, CoordinateType z) : x(x), y(y), z(z), point{x, y, z}
        {
        }

        CoordinateType x, y, z;
        CoordinateType point[3];
    };

    // Container for points
    std::vector<Point> pts;

    /// @brief Return the number of points in the cloud
    inline std::size_t kdtree_get_point_count() const noexcept
    {
        return pts.size();
    }

    /// @brief Get a point along the specified dimension
    inline CoordinateType kdtree_get_pt(const std::size_t idx, const std::size_t dim) const
    {
        switch (dim)
        {
        case (0): {
            return pts[idx].x;
        }
        case (1): {
            return pts[idx].y;
        }
        default: {
            return pts[idx].z;
        }
        }
    }

    /// @brief Optional bounding box computation
    template <class Bbox> inline bool kdtree_get_bbox(Bbox & /* bb */) const noexcept
    {
        return false;
    }
};
} // namespace clustering

#endif // POINT_STRUCT_HPP