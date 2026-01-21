#ifndef SEMANTIC_SEGMENTATION_LAYER_HPP_
#define SEMANTIC_SEGMENTATION_LAYER_HPP_

#include <functional>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"

#include "message_filters/subscriber.hpp"
#include "message_filters/time_synchronizer.hpp"
#include "nav2_costmap_2d/costmap_layer.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "semantic_segmentation_layer/segmentation_buffer.hpp"
#include "nav2_util/node_utils.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2_ros/message_filter.hpp"
#include "vision_msgs/msg/label_info.hpp"

namespace semantic_segmentation_layer {
/**
 * @class SemanticSegmentationLayer
 * @brief Takes in semantic segmentation messages and aligned pointclouds to populate the 2D costmap
 */
class SemanticSegmentationLayer : public nav2_costmap_2d::CostmapLayer
{
   public:
    /**
     * @brief A constructor
     */
    SemanticSegmentationLayer();

    /**
     * @brief A destructor
     */
    virtual ~SemanticSegmentationLayer() {}

    /**
     * @brief Initialization process of layer on startup
     */
    virtual void onInitialize();
    /**
     * @brief Update the bounds of the master costmap by this layer's update dimensions. 
     * This method includes temporal consistency by purging old observations
     * before calculating costs, ensuring the costmap reflects the current state
     * after decay time has been applied.
     * @param robot_x X pose of robot
     * @param robot_y Y pose of robot
     * @param robot_yaw Robot orientation
     * @param min_x X min map coord of the window to update
     * @param min_y Y min map coord of the window to update
     * @param max_x X max map coord of the window to update
     * @param max_y Y max map coord of the window to update
     */
    virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x, double* min_y,
                              double* max_x, double* max_y);
    /**
     * @brief Update the costs in the master costmap in the window
     * @param master_grid The master costmap grid to update
     * @param min_x X min map coord of the window to update
     * @param min_y Y min map coord of the window to update
     * @param max_x X max map coord of the window to update
     * @param max_y Y max map coord of the window to update
     */
    virtual void updateCosts(nav2_costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j);

    /**
     * @brief Reset this costmap
     */
    virtual void reset();

    virtual void onFootprintChanged();

    /**
     * @brief If clearing operations should be processed on this layer or not
     */
    virtual bool isClearable() { return true; }

    /**
     * @brief Activate this layer - subscribe to topics
     */
    virtual void activate();

    /**
     * @brief Deactivate this layer - unsubscribe from topics
     */
    virtual void deactivate();

    /**
     * @brief Get the buffers and the tile maps the plugin stores. one for each source. Takes a vector of tile maps
     * as reference and fills it inside the function
     * @param segmentation_tile_maps the vector of tile maps to be filled by the function
     * @return whether the tile maps could be retrieved and filled successfully
     */
    bool getSegmentationTileMaps(std::vector<std::pair<SegmentationTileMap::SharedPtr, SegmentationBuffer::SharedPtr>>& segmentation_tile_maps);

    rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters);

   private:
    void syncSegmPointcloudCb(const std::shared_ptr<const sensor_msgs::msg::Image>& segmentation,
                              const std::shared_ptr<const sensor_msgs::msg::PointCloud2>& pointcloud,
                              const std::shared_ptr<semantic_segmentation_layer::SegmentationBuffer>& buffer);

    void syncSegmConfPointcloudCb(const std::shared_ptr<const sensor_msgs::msg::Image>& segmentation,
                                  const std::shared_ptr<const sensor_msgs::msg::Image>& confidence,
                                  const std::shared_ptr<const sensor_msgs::msg::PointCloud2>& pointcloud,
                                  const std::shared_ptr<semantic_segmentation_layer::SegmentationBuffer>& buffer);

    void labelinfoCb(const std::shared_ptr<const vision_msgs::msg::LabelInfo>& label_info,
                     const std::shared_ptr<semantic_segmentation_layer::SegmentationBuffer>& buffer);

    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>>>
        semantic_segmentation_subs_;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>>>
        semantic_segmentation_confidence_subs_;
    std::vector<std::shared_ptr<message_filters::Subscriber<vision_msgs::msg::LabelInfo, rclcpp_lifecycle::LifecycleNode>>>
        label_info_subs_;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2, rclcpp_lifecycle::LifecycleNode>>>
        pointcloud_subs_;
    std::vector<
        std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2>>>
        segm_pc_notifiers_;
    std::vector<
        std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2>>>
        segm_conf_pc_notifiers_;
    std::vector<std::shared_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>>> pointcloud_tf_subs_;

    // debug publishers
    std::map<std::string, std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>>> proc_pointcloud_pubs_map_;

    std::vector<std::shared_ptr<semantic_segmentation_layer::SegmentationBuffer>> segmentation_buffers_;

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr dyn_params_handler_;

    std::string global_frame_;
    std::string topics_string_;

    std::map<std::string, uint8_t> class_map_;

    bool rolling_window_;
    bool was_reset_;
    int combination_method_;
};

}  // namespace semantic_segmentation_layer

#endif  // SEMANTIC_SEGMENTATION_LAYER_HPP_