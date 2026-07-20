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

#ifndef SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_BUFFER_HPP_
#define SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_BUFFER_HPP_

#include <list>
#include <algorithm>
#include <cmath>
#include <utility>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "nav2_ros_common/lifecycle_node.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.hpp"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "vision_msgs/msg/label_info.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "ground_plane_fov_checker.hpp"
#include "segmentation_cost_multimap.hpp"
#include "segmentation_tile_map.hpp"

namespace semantic_segmentation_layer {
/**
 * @class SegmentationBuffer
 * @brief Takes in point clouds from sensors, transforms them to the desired frame, and stores them
 */
class SegmentationBuffer
{
   public:
    using SharedPtr = std::shared_ptr<SegmentationBuffer>;
    /**
     * @brief  Constructs an segmentation buffer
     * @param  topic_name The topic of the segmentations, used as an identifier for error and warning
     * messages
     * @param  observation_keep_time Defines the persistence of segmentations in seconds, 0 means only
     * keep the latest
     * @param  expected_update_rate How often this buffer is expected to be updated, 0 means there is
     * no limit
     * @param  min_obstacle_height The minimum height of a hitpoint to be considered legal
     * @param  max_obstacle_height The minimum height of a hitpoint to be considered legal
     * @param  obstacle_max_range The range to which the sensor should be trusted for inserting
     * obstacles
     * @param  obstacle_min_range The range from which the sensor should be trusted for inserting
     * obstacles
     * @param  raytrace_max_range The range to which the sensor should be trusted for raytracing to
     * clear out space
     * @param  raytrace_min_range The range from which the sensor should be trusted for raytracing to
     * clear out space
     * @param  tf2_buffer A reference to a tf2 Buffer
     * @param  global_frame The frame to transform PointClouds into
     * @param  sensor_frame The frame of the origin of the sensor, can be left blank to be read from
     * the messages
     * @param  tf_tolerance The amount of time to wait for a transform to be available when setting a
     * new global frame
     */
    SegmentationBuffer(const nav2::LifecycleNode::WeakPtr& parent, std::string buffer_source,
                       std::vector<std::string> class_types,
                       std::unordered_map<std::string, CostHeuristicParams> class_names_cost_map,
                       std::unordered_map<std::string, std::vector<std::string>> class_type_to_names,
                       double observation_keep_time,
                       double expected_update_rate, double max_lookahead_distance, double min_lookahead_distance,
                       tf2_ros::Buffer& tf2_buffer, std::string global_frame, std::string sensor_frame,
                       tf2::Duration tf_tolerance, double costmap_resolution, double tile_map_decay_time, bool visualize_tile_map,
                       bool use_cost_selection,
                       double camera_h_fov, double camera_v_fov,
                       double fov_inside_decay_time, double fov_outside_decay_time, bool visualize_frustum_fov);

    /**
     * @brief  Destructor... cleans up
     */
    ~SegmentationBuffer();

    /**
     * @brief  Transforms a PointCloud to the global frame and buffers it
     * This function processes semantic segmentation data and stores observations in tiles.
     * When multiple observations exist for the same tile, the observation with the highest
     * max_cost is selected. This ensures that dangerous areas (high max_cost) are prioritized
     * over safe areas (low max_cost) for navigation safety.
     * <b>Note: The burden is on the user to make sure the transform is available... ie they should
     * use a MessageNotifier</b>
     * @param  cloud The cloud to be buffered
     * @param  segmentation The semantic segmentation image containing class IDs
     * @param  confidence The confidence image containing confidence values for each pixel
     */
    void bufferSegmentation(const sensor_msgs::msg::PointCloud2& cloud, const sensor_msgs::msg::Image& segmentation,
                            const sensor_msgs::msg::Image& confidence);

    /**
     * @brief  gets the class map associated with the segmentations stored in the buffer
     * @return the class map
     */
    std::unordered_map<std::string, CostHeuristicParams> getClassMap();

    /**
     * @brief Store the metadata of classes from label info published by the inference node
     *
     * @param label_info
     */
    void createSegmentationCostMultimap(const vision_msgs::msg::LabelInfo& label_info);

    /**
     * @brief Check if there's no class metadata stored
     *
     * @return true
     * @return false
     */
    bool isClassIdCostMapEmpty() { return segmentation_cost_multimap_->empty(); }

    /**
     * @brief  Check if the segmentation buffer is being update at its expected rate
     * @return True if it is being updated at the expected rate, false otherwise
     */
    bool isCurrent() const;

    /**
     * @brief  Lock the segmentation buffer
     */
    inline void lock() { lock_.lock(); }

    /**
     * @brief  Lock the segmentation buffer
     */
    inline void unlock() { lock_.unlock(); }

    /**
     * @brief Reset last updated timestamp
     */
    void resetLastUpdated();

    /**
     * @brief Reset last updated timestamp
     */
    std::string getBufferSource() { return buffer_source_; }

    /**
     * @brief Obtain class names strings
     *
     * @return std::vector<std::string>
     */
    std::vector<std::string> getClassTypes() { return class_types_; }

    /**
     * @brief Get class names for a specific class type
     * @param class_type The class type to get names for
     * @return Vector of class names for the given type
     */
    std::vector<std::string> getClassNamesForType(const std::string& class_type);

    void setMinObstacleDistance(double distance) { sq_min_lookahead_distance_ = pow(distance, 2); }

    void setMaxObstacleDistance(double distance) { sq_max_lookahead_distance_ = pow(distance, 2); }

    /**
     * @brief Store new class metadata when new class types are published
     *
     * @param new_class
     * @param new_cost
     */
    void updateClassMap(std::string new_class, CostHeuristicParams new_cost);

    SegmentationTileMap::SharedPtr getSegmentationTileMap()
    {
        return temporal_tile_map_;
    }

    /**
     * @brief Obtain class metadata, especially the associated cost values for a given class ID
     *
     * @param class_id
     * @return CostHeuristicParams
     */
    CostHeuristicParams getCostForClassId(uint8_t class_id)
    {
        return segmentation_cost_multimap_->getCostById(class_id);
    }

    /**
     * @brief Obtain class metadata, especially the associated cost values for a given class name
     *
     * @param class_name
     * @return CostHeuristicParams
     */
    CostHeuristicParams getCostForClassName(std::string class_name)
    {
        return segmentation_cost_multimap_->getCostByName(class_name);
    }

   private:
    /**
     * @brief  Removes any stale segmentations from the buffer list
     */
    void purgeStaleSegmentations();

    rclcpp::Clock::SharedPtr clock_;
    rclcpp::Logger logger_{rclcpp::get_logger("nav2_costmap_2d")};
    tf2_ros::Buffer& tf2_buffer_;

    bool visualize_tile_map_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tile_map_pub_;
    bool visualize_frustum_fov_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr frustum_fov_pub_;

    std::recursive_mutex lock_;  ///< @brief A lock for accessing data in callbacks safely

    std::string global_frame_;
    std::string sensor_frame_;
    std::string buffer_source_;
    tf2::Duration tf_tolerance_;

    std::vector<std::string> class_types_;
    std::unordered_map<std::string, CostHeuristicParams> class_names_cost_map_;
    std::unordered_map<std::string, std::vector<std::string>> class_type_to_names_;

    const rclcpp::Duration observation_keep_time_;
    const rclcpp::Duration expected_update_rate_;
    rclcpp::Time last_updated_;

    double sq_max_lookahead_distance_;
    double sq_min_lookahead_distance_;

    // If true, select observation per tile using highest max_cost. If false, use highest confidence
    bool use_cost_selection_ = true;

    SegmentationCostMultimap::SharedPtr segmentation_cost_multimap_;
    SegmentationTileMap::SharedPtr temporal_tile_map_;

    double camera_h_fov_;
    double camera_v_fov_;
    double fov_inside_decay_time_;
    double fov_outside_decay_time_;

    GroundPlaneFOVChecker ground_fov_checker_;
};
}  // namespace semantic_segmentation_layer
#endif  // SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_BUFFER_HPP_

