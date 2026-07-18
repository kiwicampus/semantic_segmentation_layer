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

#ifndef SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_COST_MULTIMAP_HPP_
#define SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_COST_MULTIMAP_HPP_

#include <memory>
#include <unordered_map>

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
 * Manages segmentation class information, including mapping between class names and IDs,
 * as well as managing the cost heuristic parameters associated with each class.
 */
class SegmentationCostMultimap 
{
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
                             const std::unordered_map<std::string, CostHeuristicParams>& nameToCostMap)
    {
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
    void updateCostById(uint8_t id, const CostHeuristicParams& cost)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        id_to_cost_[id] = cost;
    }

    /**
     * Retrieves the cost heuristic parameters associated with a class ID.
     * 
     * @param id The class ID.
     * @return The CostHeuristicParams associated with the class.
     */
    CostHeuristicParams getCostById(uint8_t id) const
    {
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
    bool hasClassId(uint8_t id) const
    {
        // No lock needed - only reading, no concurrent modifications
        return id_to_cost_.find(id) != id_to_cost_.end();
    }

    /**
     * Updates the cost heuristic parameters associated with a class name.
     * 
     * @param name The class name.
     * @param cost The new CostHeuristicParams to associate with the class.
     */
    void updateCostByName(const std::string& name, const CostHeuristicParams& cost)
    {
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
    CostHeuristicParams getCostByName(const std::string& name) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        uint8_t id = name_to_id_.at(name);
        return id_to_cost_.at(id);
    }

    bool empty()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return name_to_id_.empty() || id_to_cost_.empty();
    }

private:
    mutable std::mutex mutex_;  // mutable allows locking in const methods
    std::unordered_map<std::string, uint8_t> name_to_id_;
    std::unordered_map<uint8_t, CostHeuristicParams> id_to_cost_;
};

#endif  // SEMANTIC_SEGMENTATION_LAYER__SEGMENTATION_COST_MULTIMAP_HPP_