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

#include "semantic_segmentation_layer/temporal_observation_queue.hpp"

void TemporalObservationQueue::push(TileObservation tile_obs, bool dominant_priority)
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

float TemporalObservationQueue::getConfidenceSum() const
{
    if (dominant_class_id_ != -1)
    {
        auto it = class_confidence_sums_.find(dominant_class_id_);
        return (it != class_confidence_sums_.end()) ? it->second : 0.0f;
    }
    return 0.0f;
}

std::deque<TileObservation> TemporalObservationQueue::getQueue()
{
    if (dominant_class_id_ != -1)
    {
        auto it = class_queues_.find(dominant_class_id_);
        return (it != class_queues_.end()) ? it->second : std::deque<TileObservation>();
    }
    return std::deque<TileObservation>();
}

void TemporalObservationQueue::purgeOld(double current_time)
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

void TemporalObservationQueue::clearQueuesExcept(uint8_t keep_class_id)
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

void TemporalObservationQueue::recomputeDominant()
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