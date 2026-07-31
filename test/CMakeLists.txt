# Test helper functions
nav2_add_gtest(test_utils
  test_utils.cpp
)
target_link_libraries(test_utils
  semantic_segmentation_layer
)

# Test temporal observation queue
nav2_add_gtest(test_temporal_observation_queue
  test_temporal_observation_queue.cpp
)
target_link_libraries(test_temporal_observation_queue
  semantic_segmentation_layer
)

# Test segmentation tile map
nav2_add_gtest(test_segmentation_tile_map
  test_segmentation_tile_map.cpp
)
target_link_libraries(test_segmentation_tile_map
  semantic_segmentation_layer
)
