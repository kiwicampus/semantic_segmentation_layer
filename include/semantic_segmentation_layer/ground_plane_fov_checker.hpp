/*********************************************************************
 *
 * Software License Agreement
 *
 *  Copyright (c) 2026, robot.com
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of robot.com nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Pedro Gonzalez (pedro@robot.com)
 *          Johan Solarte (jsolarte@robot.com)
 *********************************************************************/

#ifndef SEMANTIC_SEGMENTATION_LAYER__GROUND_PLANE_FOV_CHECKER_HPP_
#define SEMANTIC_SEGMENTATION_LAYER__GROUND_PLANE_FOV_CHECKER_HPP_

#include <Eigen/Geometry>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

/**
 * @brief 2D ground-plane FOV checker for points at z=0.
 * Four corner rays → footprint on z=0. Along each ray, distance is capped by an effective max range:
 * if both upper corner rays (sv=+1 in local build order) hit z=0, that cap is
 * min(max_lookahead_distance, min(t_upper0, t_upper1)); otherwise max_lookahead_distance.
 * Along each ray: use z=0 intersection when valid, else clamp at frustum_end_dist.
 */
class GroundPlaneFOVChecker
{
public:
    GroundPlaneFOVChecker(double hFOV, double vFOV, double max_range)
        : hFOV_(hFOV), vFOV_(vFOV), max_range_(max_range)
    {
        buildLocalRays();
    }

    void updatePose(const geometry_msgs::msg::Point& pos, const geometry_msgs::msg::Quaternion& quat)
    {
        position_ = Eigen::Vector3d(pos.x, pos.y, pos.z);
        orientation_ = Eigen::Quaterniond(quat.w, quat.x, quat.y, quat.z);
        orientation_.normalize();
        recomputeGroundPolygon();
    }

    bool isInFOV(double wx, double wy) const
    {
        if (ground_polygon_.size() < 3) return false;
        return pointInPolygon(Vec2D{wx, wy}, ground_polygon_);
    }

    /** @return Ground polygon as points (z=0) for visualization; empty if invalid. */
    std::vector<geometry_msgs::msg::Point> getGroundPolygonForVisualization() const
    {
        std::vector<geometry_msgs::msg::Point> out;
        out.reserve(ground_polygon_.size());
        for (const auto& p : ground_polygon_) {
            geometry_msgs::msg::Point pt;
            pt.x = p.x;
            pt.y = p.y;
            pt.z = 0.0;
            out.push_back(pt);
        }
        return out;
    }

private:
    struct Vec2D { double x, y; };

    double hFOV_, vFOV_, max_range_;
    Eigen::Vector3d position_;
    Eigen::Quaterniond orientation_;
    std::vector<Eigen::Vector3d> local_rays_;
    std::vector<Vec2D> ground_polygon_;

    void buildLocalRays() // Build the four local rays that represent the four corners of the FOV
    {
        local_rays_.clear();
        Eigen::Vector3d Z = Eigen::Vector3d::UnitZ(); // Vector pointing away from the camera
        for (int sv : {1, -1}) {
            for (int sh : {1, -1}) {
                Eigen::Affine3d rx(Eigen::AngleAxisd(sv * vFOV_ / 2.0, Eigen::Vector3d::UnitX()));
                Eigen::Affine3d ry(Eigen::AngleAxisd(sh * hFOV_ / 2.0, Eigen::Vector3d::UnitY()));
                local_rays_.push_back((rx * ry * Z).normalized());
            }
        }
    }

    // Calculate the xy of the point on the ray at a given distance from the origin
    static Vec2D rayPointAtDistance(
        const Eigen::Vector3d& origin, const Eigen::Vector3d& d_unit, double dist)
    {
        Eigen::Vector3d p = origin + dist * d_unit;
        return Vec2D{p.x(), p.y()};
    }

    // Find the distance along the ray to the z = 0 plane
    static bool distanceToZ0(const Eigen::Vector3d& origin, const Eigen::Vector3d& d_unit, double& t_out)
    {
        const double eps = 1e-9;
        if (d_unit.z() >= -eps) { // The ray is parallel to the plane
            return false;
        }
        const double distance_to_z0 = -origin.z() / d_unit.z(); // Calculate the distance to the z = 0 plane
        if (distance_to_z0 <= 0.0) { // The ray hits behind the origin
            return false;
        }
        t_out = distance_to_z0;
        return true;
    }

    // Ground xy: z=0 hit when valid, else point at frustum_end_dist along the ray.
    static Vec2D groundHit(
        const Eigen::Vector3d& origin, const Eigen::Vector3d& d_unit, bool hit_z0, double dist_z0,
        double frustum_end_dist)
    {
        if (!hit_z0) { // If the ray does not hit the z = 0 plane, use the max distance
            return rayPointAtDistance(origin, d_unit, frustum_end_dist);
        }
        if (dist_z0 > frustum_end_dist) {
            return rayPointAtDistance(origin, d_unit, frustum_end_dist);
        }
        return rayPointAtDistance(origin, d_unit, dist_z0);
    }

    /** Order 4 points CCW around centroid so LINE_STRIP closes without self-intersection. */
    static std::vector<Vec2D> orderQuadCCW(std::vector<Vec2D> pts)
    {
        if (pts.size() < 3) return pts;
        double cx = 0, cy = 0;
        for (const auto& p : pts) {
            cx += p.x;
            cy += p.y;
        }
        cx /= static_cast<double>(pts.size());
        cy /= static_cast<double>(pts.size());
        std::sort(pts.begin(), pts.end(), [cx, cy](const Vec2D& a, const Vec2D& b) {
            return std::atan2(a.y - cy, a.x - cx) < std::atan2(b.y - cy, b.x - cx);
        });
        return pts;
    }

    void recomputeGroundPolygon()
    {
        if (local_rays_.size() != 4u) {
            ground_polygon_.clear();
            return;
        }
        std::vector<Eigen::Vector3d> world_dir(4);
        std::vector<bool> hit_z0(4);
        std::vector<double> dist_z0(4);
        for (size_t i = 0; i < 4u; ++i) { // Find distance to z = 0 for each ray
            world_dir[i] = (orientation_ * local_rays_[i]).normalized();
            hit_z0[i] = distanceToZ0(position_, world_dir[i], dist_z0[i]);
        }
        double frustum_end_dist = max_range_;
        if (hit_z0[0] && hit_z0[1]) { // Upper corner rays hit z = 0, use the min of the two distances
            frustum_end_dist = std::min(max_range_, std::min(dist_z0[0], dist_z0[1]));
        }
        std::vector<Vec2D> candidates;
        candidates.reserve(4);
        for (size_t i = 0; i < 4u; ++i) { // Calculate the ground hit for each ray
            candidates.push_back(groundHit(
                position_, world_dir[i], hit_z0[i], dist_z0[i], frustum_end_dist));
        }
        ground_polygon_ = orderQuadCCW(std::move(candidates));
    }

    /**
     * Point-in-polygon test.
     * Cross product (b-a)×(p-a): positive => point to left of edge => INSIDE.
     * cross == 0 (on edge) treated as inside for robustness.
     */
    static bool pointInPolygon(const Vec2D& p, const std::vector<Vec2D>& poly)
    {
        const size_t n = poly.size();
        for (size_t i = 0; i < n; ++i) {
            const Vec2D& a = poly[i];
            const Vec2D& b = poly[(i + 1) % n];
            const double cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
            if (cross < 0) return false;  // outside (right of edge)
        }
        return true;  // inside or on edge
    }
};

#endif  // SEMANTIC_SEGMENTATION_LAYER__GROUND_PLANE_FOV_CHECKER_HPP_