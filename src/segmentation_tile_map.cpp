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

#include "semantic_segmentation_layer/segmentation_tile_map.hpp"

SegmentationTileMap::SegmentationTileMap(float resolution, float decay_time)
: resolution_(resolution), decay_time_(decay_time)
{
    tile_map_.reserve(1e4);
}

TileIndex SegmentationTileMap::worldToIndex(double x, double y) const
{
    // Convert world coordinates to grid indices
    int ix = static_cast<int>(std::floor(x / resolution_));
    int iy = static_cast<int>(std::floor(y / resolution_));
    
    return TileIndex{ix, iy};
}

TileWorldXY SegmentationTileMap::indexToWorld(int x, int y) const
{
    // Calculate the world coordinates of the center of the grid cell
    double x_world = (static_cast<double>(x) + 0.5) * resolution_;
    double y_world = (static_cast<double>(y) + 0.5) * resolution_;
    
    return TileWorldXY{x_world, y_world};
}

void SegmentationTileMap::pushObservation(TileObservation& obs, TileIndex& idx, bool dominant_priority)
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

void SegmentationTileMap::purgeOldObservations(double current_time)
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
