std_msgs/Header header
int32 fruit_id
string fruit_type
string fruit_variety
string ripeness
float32 area            # Area of the detected fruit
float32[] bbox          # Bounding box as [x, y, width, height]
float32[] mask          # Mask for segmented fruit 