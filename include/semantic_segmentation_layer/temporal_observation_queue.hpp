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

#ifndef SEMANTIC_SEGMENTATION_LAYER__TEMPORAL_OBSERVATION_QUEUE_HPP_
#define SEMANTIC_SEGMENTATION_LAYER__TEMPORAL_OBSERVATION_QUEUE_HPP_

#include <memory>
#include <deque>
#include <unordered_map>

/**
 * @brief Encapsulates the observation data for a tile, including class ID, cost, confidence, and timestamp.
 */
struct TileObservation
{
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
    void push(TileObservation tile_obs, bool dominant_priority = false);

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
    float getConfidenceSum() const;

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
    std::deque<TileObservation> getQueue();

    /**
     * @brief Removes observations older than the decay time from all class queues.
     * @param current_time The current time for comparison.
     */
    void purgeOld(double current_time);

private:
    /**
     * @brief Removes all class queues and confidence sums except the specified class.
     * @param keep_class_id The class ID to preserve.
     */
    void clearQueuesExcept(uint8_t keep_class_id);

    /**
     * @brief Recomputes dominant_class_id_ and dominant_class_size_ by scanning class_queues_.
     */
    void recomputeDominant();

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

#endif  // SEMANTIC_SEGMENTATION_LAYER__TEMPORAL_OBSERVATION_QUEUE_HPP_
