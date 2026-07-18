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

#ifndef SEMANTIC_SEGMENTATION_LAYER__UTILS_HPP_
#define SEMANTIC_SEGMENTATION_LAYER__UTILS_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "segmentation_tile_map.hpp"

/**
 * @brief Creates a PointCloud2 message that contains a visual representation of 
 * a temporal tile map. There's a "column" of points on each tile, each point represents
 * a segmentation observation over that tile and they are all stacked together. Each observation
 * Has a channel for the class, for the confidence, and the confidence sum of the observations
 * over that tile
 * @param tileMap The segmentation tile map
 */
inline sensor_msgs::msg::PointCloud2 visualizeTemporalTileMap(SegmentationTileMap& tileMap, const std::string& frame_id,
                                                              const rclcpp::Time& stamp)
{
    /**
    * @brief Struct for holding the relevant data of any observation. Includes
    * its position, its confidence, the confidence sum of the tile and the
    * class to which it belongs
    */
    struct PointData 
    {
        float x, y, z;
        float confidence, confidence_sum;
        uint8_t class_id;
    };

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.frame_id = frame_id;
    cloud.header.stamp = stamp;

    // Define fields for PointCloud2
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(6, "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                                     "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                                     "z", 1, sensor_msgs::msg::PointField::FLOAT32,
                                     "confidence", 1, sensor_msgs::msg::PointField::FLOAT32,
                                     "confidence_sum", 1, sensor_msgs::msg::PointField::FLOAT32,
                                     "class", 1, sensor_msgs::msg::PointField::UINT8);

    // Reserve space for points
    std::vector<PointData> points;
    for (auto& tile : tileMap) {
        TileIndex idx = tile.first;
        TileWorldXY worldXY = tileMap.indexToWorld(idx.x, idx.y);
        double z = 0.0;
        for (auto& obs : tile.second.getQueue()) {
            PointData point;
            point.x = worldXY.x;
            point.y = worldXY.y;
            point.z = z;
            point.confidence = obs.confidence;
            point.confidence_sum = tile.second.getConfidenceSum() / tile.second.size();
            point.class_id = static_cast<uint8_t>(obs.class_id);
            points.push_back(point);
            z += 0.02;  // Increment Z by 0.02m for each observation
        }
    }

    // Set data in PointCloud2
    modifier.resize(points.size());  // Number of points
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_confidence(cloud, "confidence");
    sensor_msgs::PointCloud2Iterator<float> iter_confidence_sum(cloud, "confidence_sum");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_class(cloud, "class");

    for (const auto& point : points) {
        *iter_x = point.x;
        *iter_y = point.y;
        *iter_z = point.z;
        *iter_confidence = point.confidence;
        *iter_confidence_sum = point.confidence_sum;
        *iter_class = point.class_id;
        ++iter_x; ++iter_y; ++iter_z; ++iter_confidence;++iter_confidence_sum; ++iter_class;
    }

    return cloud;
}

#endif  // SEMANTIC_SEGMENTATION_LAYER__UTILS_HPP_