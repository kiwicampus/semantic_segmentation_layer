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

#ifndef SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_TILE_MAP_HPP_
#define SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_TILE_MAP_HPP_

#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "temporal_observation_queue.hpp"

/**
 * @brief Represents a 2D grid index with equality comparison. Supports negative indexes
 */
struct TileIndex 
{
    int x, y;

    bool operator==(const TileIndex& other) const {
        return x == other.x && y == other.y;
    }
};

/**
 * @brief Represents the world coordinates of a tile.
 */
struct TileWorldXY
{
    double x, y;
};

/**
* @brief Custom hash function for TileIndex to enable its use as a key in unordered_map.
*/
template<>
struct std::hash<TileIndex> 
{
    size_t operator()(const TileIndex& coord) const {
        // Compute individual hash values for two integers
        // and combine them using bitwise XOR
        // and bit shifting:
        return std::hash<int>()(coord.x) ^ (std::hash<int>()(coord.y) << 1);
    }
};

/**
 * @brief Manages a map of tile observations, allowing for spatial and temporal querying.
 * Utilizes an unordered_map to efficiently index observations by tile and supports locking for thread safety.
 */
class SegmentationTileMap 
{
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

    SegmentationTileMap(float resolution, float decay_time);
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
    int size() { return tile_map_.size(); }
    
    /**
    * @brief Returns the decay time for queue
    * 
    * @return float 
    */
    float getDecayTime() const { return decay_time_; }

    /**
    * @brief Converts world coordinates to a TileIndex.
    * @param x X coordinate in world space.
    * @param y Y coordinate in world space.
    * @return The corresponding TileIndex.
    */
    TileIndex worldToIndex(double x, double y) const;

    /**
    * @brief Converts a TileIndex to world coordinates.
    * @param idx The index to convert.
    * @return The world coordinates of the tile's center.
    */
    TileWorldXY indexToWorld(int x, int y) const;

    /**
    * @brief Adds an observation to the specified tile.
    * @param obs The observation to add.
    * @param idx The index of the tile.
    * @param dominant_priority Whether this class should take immediate dominance when observed.
    */
    void pushObservation(TileObservation& obs, TileIndex& idx, bool dominant_priority = false);

    /**
    * @brief Removes observations older than the decay time from all tiles.
    * @param current_time The current time for comparison.
    */
    void purgeOldObservations(double current_time);
};

#endif  // SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_TILE_MAP_HPP_