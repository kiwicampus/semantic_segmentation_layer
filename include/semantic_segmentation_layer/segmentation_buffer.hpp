#ifndef SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_BUFFER_HPP_
#define SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_BUFFER_HPP_

#include <list>
#include <string>
#include <vector>

#include "nav2_util/lifecycle_node.hpp"
#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.hpp"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "vision_msgs/msg/label_info.hpp"

/**
 * @brief Represents the parameters associated with the cost calculation for a given class
 */
struct CostHeuristicParams
{
    uint8_t base_cost, max_cost, mark_confidence;
    int samples_to_max_cost;
    bool dominant_priority;
};

/**
 * @brief Represents a 2D grid index with equality comparison. Supports negative indexes
 */
struct TileIndex {
    int x, y;

    bool operator==(const TileIndex& other) const {
        return x == other.x && y == other.y;
    }
};

namespace std {
    /**
     * @brief Custom hash function for TileIndex to enable its use as a key in unordered_map.
     */
    template<>
    struct hash<TileIndex> {
        size_t operator()(const TileIndex& coord) const {
            // Compute individual hash values for two integers
            // and combine them using bitwise XOR
            // and bit shifting:
            return std::hash<int>()(coord.x) ^ (std::hash<int>()(coord.y) << 1);
        }
    };
}


/**
 * @brief Represents the world coordinates of a tile.
 */
struct TileWorldXY
{
    double x, y;
};

/**
 * @brief Encapsulates the observation data for a tile, including class ID, cost, confidence, and timestamp.
 */
struct TileObservation {
    using UniquePtr = std::unique_ptr<TileObservation>;

    uint8_t class_id;
    float confidence;
    double timestamp;
};

/**
 * @brief Manages temporal observations with a decay mechanism, maintaining a sum of confidences.
 * Wraps multiple std::deque objects to store observations per class ID, allowing for efficient insertion and removal.
 * Uses class ID -1 as a sentinel value to indicate no dominant class exists.
 */
class TemporalObservationQueue
{
   private:
    std::unordered_map<uint8_t, std::deque<TileObservation>> class_queues_;
    std::unordered_map<uint8_t, float> class_confidence_sums_;
    int dominant_class_id_ = -1;
    size_t dominant_class_size_ = 0;
    double decay_time_;

   public:
    TemporalObservationQueue() {}

    /**
     * @brief Adds an observation to the appropriate class queue, manages dominant class tracking.
     * @param tile_obs The observation to add.
     * @param dominant_priority Whether this class should take immediate dominance when observed.
     */
    void push(TileObservation tile_obs, bool dominant_priority = false)
    {
        uint8_t class_id = tile_obs.class_id;
        
        // Add observation to the appropriate class queue
        auto& queue = class_queues_[class_id];
        queue.push_back(tile_obs);
        
        // Update confidence sum for this class
        class_confidence_sums_[class_id] += tile_obs.confidence;
        
        // Check if this class should become dominant
        size_t current_class_size = queue.size();
        bool should_become_dominant = false;
        
        if (dominant_priority) {
            should_become_dominant = true;
        } else {
            //logic for non-dominant_priority classes: only compete by size
            should_become_dominant = (current_class_size > dominant_class_size_);
        }
        
        if (should_become_dominant)
        {
            // New dominant class - purge all other classes
            if (dominant_class_id_ != -1 && dominant_class_id_ != class_id)
            {
                clearQueuesExcept(class_id);
            }
            
            // Update dominance
            setDominant(class_id, current_class_size);
        }
    }

    /**
     * @brief Checks if the dominant class queue is empty.
     * @return True if empty, false otherwise.
     */
    bool empty() const { return dominant_class_id_ == -1; }

    /**
     * @brief Gets the size of the dominant class queue.
     * @return The number of observations in the dominant class queue.
     */
    size_t size() const { return dominant_class_size_; }

    /**
     * @brief Sets the decay time for observations.
     * @param decay_time The decay time in seconds.
     */
    void setDecayTime(float decay_time) { decay_time_ = decay_time; }

    /**
     * @brief Gets the current sum of confidence values of the dominant class.
     * @return The sum of confidences for the dominant class.
     */
    float getConfidenceSum() const 
    { 
        if (dominant_class_id_ != -1)
        {
            auto it = class_confidence_sums_.find(dominant_class_id_);
            return (it != class_confidence_sums_.end()) ? it->second : 0.0f;
        }
        return 0.0f;
    }

    /**
     * @brief Gets the class ID of the dominant class (most samples).
     * @return The class ID, or -1 if no observations exist (-1 is used as sentinel value).
     */
    int getClassId() const { return dominant_class_id_; }

    /**
     * @brief Returns a copy of the dominant class queue. Will have overhead
     * due to the copy operation but avoids race conditions since
     * the object in the class is not made editable by others
     * @return The dominant class queue, or empty deque if no dominant class.
     */
    std::deque<TileObservation> getQueue() 
    { 
        if (dominant_class_id_ != -1)
        {
            auto it = class_queues_.find(dominant_class_id_);
            return (it != class_queues_.end()) ? it->second : std::deque<TileObservation>();
        }
        return std::deque<TileObservation>();
    }

    /**
     * @brief Removes observations older than the decay time from all class queues.
     * @param current_time The current time for comparison.
     */
    void purgeOld(double current_time)
    {
        // Iterate through all class queues and remove time-expired observations.
        // While doing so, maintain the running confidence sums and remove classes
        // whose queues become empty to preserve the invariant: if a class exists
        // in class_queues_, its queue size is >= 1.
        bool dominant_removed = false;

        for (auto it = class_queues_.begin(); it != class_queues_.end(); )
        {
            auto& queue = it->second;
            const uint8_t class_id = it->first;
        
            // Pop observations older than decay_time_ from the front (oldest first),
            // updating the confidence sum accordingly.
            while (!queue.empty())
            {
                double age = current_time - queue.front().timestamp;
                if (age > decay_time_)
                {
                    class_confidence_sums_[class_id] -= queue.front().confidence;
                    queue.pop_front();
                }
                else
                {
                    break;
                }
            }
        
            // If the queue ended up empty, erase the class entry entirely to avoid
            // keeping "zombie" keys and to keep class_queues_ and class_confidence_sums_
            // in sync. Track if the dominant class was removed to recompute dominance later.
            if (queue.empty())
            {
                if (class_id == dominant_class_id_) dominant_removed = true;
                class_confidence_sums_.erase(class_id);
                it = class_queues_.erase(it);
            }
            else
            {
                ++it;
            }
        }
        
        // Update dominant class bookkeeping:
        // - If the dominant class was removed, scan to find the new dominant.
        // - Otherwise, just refresh the dominant_class_size_ if it still exists;
        //   if not found (edge case), reset dominance.
        if (dominant_removed) {
            recomputeDominant();
        } else if (dominant_class_id_ != -1) {
            auto it = class_queues_.find(dominant_class_id_);
            if (it != class_queues_.end()) setDominant(dominant_class_id_, it->second.size());
            else resetDominant();
        }
    }

private:
    /**
     * @brief Removes all class queues and confidence sums except the specified class.
     * @param keep_class_id The class ID to preserve.
     */
    void clearQueuesExcept(uint8_t keep_class_id)
    {
        for (auto it = class_queues_.begin(); it != class_queues_.end();)
        {
            if (it->first != keep_class_id)
            {
                class_confidence_sums_.erase(it->first);
                it = class_queues_.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    /**
     * @brief Recomputes dominant_class_id_ and dominant_class_size_ by scanning class_queues_.
     */
    void recomputeDominant()
    {
        resetDominant();
        for (const auto& pair : class_queues_)
        {
            if (pair.second.size() > dominant_class_size_)
            {
                setDominant(pair.first, pair.second.size());
            }
        }
    }

    /**
     * @brief Resets the dominant class state to none.
     */
    void resetDominant()
    {
        dominant_class_id_ = -1;
        dominant_class_size_ = 0;
    }

    /**
     * @brief Sets the dominant class and its current size.
     */
    void setDominant(uint8_t class_id, size_t size)
    {
        dominant_class_id_ = class_id;
        dominant_class_size_ = size;
    }
};

/**
 * @brief Manages a map of tile observations, allowing for spatial and temporal querying.
 * Utilizes an unordered_map to efficiently index observations by tile and supports locking for thread safety.
 */
class SegmentationTileMap {
    private:
        std::unordered_map<TileIndex, TemporalObservationQueue> tile_map_;
        float resolution_;
        float decay_time_;
        std::recursive_mutex lock_;


    public:
        using SharedPtr = std::shared_ptr<SegmentationTileMap>;

        // Define iterator types
        using Iterator = typename std::unordered_map<TileIndex, TemporalObservationQueue>::iterator;
        using ConstIterator = typename std::unordered_map<TileIndex, TemporalObservationQueue>::const_iterator;

        SegmentationTileMap(float resolution, float decay_time) : resolution_(resolution), decay_time_(decay_time) {
            // 10k observations seemed to be a good estimate of the amount of data to be held for a decay time of ~5s
            tile_map_.reserve(1e4);
        }
        SegmentationTileMap(){}

        // Return iterator to the beginning of the tile_map_
        Iterator begin() { return tile_map_.begin(); }
        ConstIterator begin() const { return tile_map_.begin(); }

        // Return iterator to the end of the tile_map_
        Iterator end() { return tile_map_.end(); }
        ConstIterator end() const { return tile_map_.end(); }

        /**
         * @brief Locks the map for exclusive access.
         */
        inline void lock() { lock_.lock(); }

        /**
         * @brief Unlocks the map.
         */
        inline void unlock() { lock_.unlock(); }

        /**
         * @brief Returns the number of elements in the map.
         * @return The size of the map.
         */
        int size()
        {
            return tile_map_.size();
        }

        /**
         * @brief Converts world coordinates to a TileIndex.
         * @param x X coordinate in world space.
         * @param y Y coordinate in world space.
         * @return The corresponding TileIndex.
         */
        TileIndex worldToIndex(double x, double y) const {
            // Convert world coordinates to grid indices
            int ix = static_cast<int>(std::floor(x / resolution_));
            int iy = static_cast<int>(std::floor(y / resolution_));
            return TileIndex{ix, iy};
        }

        /**
         * @brief Converts a TileIndex to world coordinates.
         * @param idx The index to convert.
         * @return The world coordinates of the tile's center.
         */
        TileWorldXY indexToWorld(int x, int y) const {
            // Calculate the world coordinates of the center of the grid cell
            double x_world = (static_cast<double>(x) + 0.5) * resolution_;
            double y_world = (static_cast<double>(y) + 0.5) * resolution_;
            return TileWorldXY{x_world, y_world};
        }

        /**
         * @brief Adds an observation to the specified tile.
         * @param obs The observation to add.
         * @param idx The index of the tile.
         * @param dominant_priority Whether this class should take immediate dominance when observed.
         */
        void pushObservation(TileObservation& obs, TileIndex& idx, bool dominant_priority = false)
        {
            auto it = tile_map_.find(idx);
            if (it != tile_map_.end())
            {
                // TileIndex exists, push the observation with dominance flag
                it->second.push(obs, dominant_priority);
            }
            else
            {
                // TileIndex does not exist, create a new TemporalObservationQueue with decay time
                TemporalObservationQueue& queue = tile_map_[idx];
                queue.setDecayTime(decay_time_);
                queue.push(obs, dominant_priority);
            }
        }

        /**
         * @brief Removes observations older than the decay time from all tiles.
         * @param current_time The current time for comparison.
         */
        void purgeOldObservations(double current_time)
        {
            std::vector<TileIndex> tiles_to_remove;
            for (auto& tile : tile_map_)
            {
                tile.second.purgeOld(current_time);
                if(tile.second.empty())
                {
                    tiles_to_remove.emplace_back(tile.first);
                }
            }
            if(tile_map_.size() > 0)
            for (auto& tile : tiles_to_remove)
            {
                tile_map_.erase(tile);
            }
        }
};

/**
 * @brief Struct for holding the relevant data of any observation. Includes
 * its position, its confidence, the confidence sum of the tile and the
 * class to which it belongs
 */
struct PointData {
    float x, y, z;
    float confidence, confidence_sum;
    uint8_t class_id;
};

/**
 * @brief Creates a PointCloud2 message that contains a visual representation of 
 * a temporal tile map. There's a "column" of points on each tile, each point represents
 * a segmentation observation over that tile and they are all stacked together. Each observation
 * Has a channel for the class, for the confidence, and the confidence sum of the observations
 * over that tile
 * @param tileMap The segmentation tile map
 */
inline sensor_msgs::msg::PointCloud2 visualizeTemporalTileMap(SegmentationTileMap& tileMap) {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.frame_id = "map";  // Set appropriate frame_id
    cloud.header.stamp = rclcpp::Clock().now();  // Set current time as timestamp

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

/**
 * Manages segmentation class information, including mapping between class names and IDs,
 * as well as managing the cost heuristic parameters associated with each class.
 */
class SegmentationCostMultimap {
public:
    using SharedPtr = std::shared_ptr<SegmentationCostMultimap>;
    SegmentationCostMultimap(){}
    /**
     * Constructs the SegmentationCostMultimap.
     * 
     * @param nameToIdMap A map from class names to class IDs.
     * @param nameToCostMap A map from class names to CostHeuristicParams.
     */
    SegmentationCostMultimap(const std::unordered_map<std::string, uint8_t>& nameToIdMap,
                             const std::unordered_map<std::string, CostHeuristicParams>& nameToCostMap) {
        std::lock_guard<std::mutex> lock(mutex_);
        name_to_id_ = nameToIdMap;
        for (const auto& pair : nameToIdMap) {
            const auto& name = pair.first;
            uint8_t id = pair.second;
            auto cost_it = nameToCostMap.find(name);
            if (cost_it == nameToCostMap.end()) {
                // This shouldn't happen because we already checked in createSegmentationCostMultimap
                // but let's be extra safe
                id_to_cost_[id] = CostHeuristicParams{0, 0, 0, 0, false};
                continue;
            }
            id_to_cost_[id] = cost_it->second;
        }
    }

    /**
     * Updates the cost heuristic parameters associated with a class ID.
     * 
     * @param id The class ID.
     * @param cost The new CostHeuristicParams to associate with the class.
     */
    void updateCostById(uint8_t id, const CostHeuristicParams& cost) {
        std::lock_guard<std::mutex> lock(mutex_);
        id_to_cost_[id] = cost;
    }

    /**
     * Retrieves the cost heuristic parameters associated with a class ID.
     * 
     * @param id The class ID.
     * @return The CostHeuristicParams associated with the class.
     */
    CostHeuristicParams getCostById(uint8_t id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = id_to_cost_.find(id);
        if (it == id_to_cost_.end()) {
            return CostHeuristicParams{0, 0, 0, 0, false};
        }
        return it->second;
    }

    /**
     * Checks if a class ID exists in the cost mapping.
     * 
     * @param id The class ID to check.
     * @return true if the class ID exists, false otherwise.
     */
    bool hasClassId(uint8_t id) const {
        // No lock needed - only reading, no concurrent modifications
        return id_to_cost_.find(id) != id_to_cost_.end();
    }

    /**
     * Updates the cost heuristic parameters associated with a class name.
     * 
     * @param name The class name.
     * @param cost The new CostHeuristicParams to associate with the class.
     */
    void updateCostByName(const std::string& name, const CostHeuristicParams& cost) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint8_t id = name_to_id_.at(name);
        id_to_cost_[id] = cost;
    }

    /**
     * Retrieves the cost heuristic parameters associated with a class name.
     * 
     * @param name The class name.
     * @return The CostHeuristicParams associated with the class.
     */
    CostHeuristicParams getCostByName(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        uint8_t id = name_to_id_.at(name);
        return id_to_cost_.at(id);
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return name_to_id_.empty() || id_to_cost_.empty();
    }

private:
    mutable std::mutex mutex_;  // mutable allows locking in const methods
    std::unordered_map<std::string, uint8_t> name_to_id_;
    std::unordered_map<uint8_t, CostHeuristicParams> id_to_cost_;
};

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
    SegmentationBuffer(const nav2_util::LifecycleNode::WeakPtr& parent, std::string buffer_source,
                       std::vector<std::string> class_types,
                       std::unordered_map<std::string, CostHeuristicParams> class_names_cost_map,
                       std::unordered_map<std::string, std::vector<std::string>> class_type_to_names,
                       double observation_keep_time,
                       double expected_update_rate, double max_lookahead_distance, double min_lookahead_distance,
                       tf2_ros::Buffer& tf2_buffer, std::string global_frame, std::string sensor_frame,
                       tf2::Duration tf_tolerance, double costmap_resolution, double tile_map_decay_time, bool visualize_tile_map = false,
                       bool use_cost_selection = true);

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

    void createSegmentationCostMultimap(const vision_msgs::msg::LabelInfo& label_info);

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
    std::vector<std::string> getClassTypes() { return class_types_; }
    
    /**
     * @brief Get class names for a specific class type
     * @param class_type The class type to get names for
     * @return Vector of class names for the given type
     */
    std::vector<std::string> getClassNamesForType(const std::string& class_type);

    void setMinObstacleDistance(double distance) { sq_min_lookahead_distance_ = pow(distance, 2); }

    void setMaxObstacleDistance(double distance) { sq_max_lookahead_distance_ = pow(distance, 2); }

    void updateClassMap(std::string new_class, CostHeuristicParams new_cost);

    SegmentationTileMap::SharedPtr getSegmentationTileMap()
    {
        return temporal_tile_map_;
    }

    CostHeuristicParams getCostForClassId(uint8_t class_id)
    {
        return segmentation_cost_multimap_->getCostById(class_id);
    }

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
    std::vector<std::string> class_types_;
    std::unordered_map<std::string, CostHeuristicParams> class_names_cost_map_;
    std::unordered_map<std::string, std::vector<std::string>> class_type_to_names_;
    const rclcpp::Duration observation_keep_time_;
    const rclcpp::Duration expected_update_rate_;
    rclcpp::Time last_updated_;
    std::string global_frame_;
    std::string sensor_frame_;
    std::string buffer_source_;
    std::recursive_mutex lock_;  ///< @brief A lock for accessing data in callbacks safely
    double sq_max_lookahead_distance_;
    double sq_min_lookahead_distance_;
    tf2::Duration tf_tolerance_;
    
    SegmentationCostMultimap::SharedPtr segmentation_cost_multimap_;

    SegmentationTileMap::SharedPtr temporal_tile_map_;

    bool visualize_tile_map_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tile_map_pub_;
    // If true, select observation per tile using highest max_cost. If false, use highest confidence
    bool use_cost_selection_ = true;
};
}  // namespace semantic_segmentation_layer
#endif  // SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_BUFFER_HPP_

