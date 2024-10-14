# FruitInfoArray.msg
aoc_fruit_detector/FruitInfoMessage[] fruits    # Detected fruits on the image
sensor_msgs/Image rgb_image                     # RGB image used for detection
sensor_msgs/Image depth_image                   # Depth image used for detection
sensor_msgs/Image rgb_image_composed            # RGB image composed with detected fruits