# FruitInfoMessage.msg

std_msgs/Header header
int64 image_id              # ID of the image that fruit belongs to
int32 fruit_id              # ID of the detected fruit/vegetable
string fruit_family         # Biological family of the fruit (e.g., "Rosaceae")
string fruit_species        # Species name (e.g., "Malus domestica")
string fruit_type           # General botanical type (e.g., "Apple", "Tomato")
string fruit_variety        # Specific variety or cultivar (e.g., "Granny Smith")
string fruit_genotype       # Genotype of the fruit (useful for research or breeding)
string fruit_quality        # Quality measure (e.g., "High", "Medium", "Low", "Rotten")
string ripeness_category    # Ripeness category (e.g., "Unripe", "Ripe", "Overripe")
float32 ripeness_level      # Ripeness level between 0 and 1, continuous metric for ripeness, more than 1 is overripe
float32 area                # Area of the detected fruit/vegetable
float32 volume              # Volume of the detected fruit/vegetable
float32 weight              # Weight of the detected fruit/vegetable
float32 fruit_stem_length   # Length of the fruit's stem
float32[] bbox              # Bounding box as [x, y, width, height]
float32[] bvol              # Bounding volume as [x, y, z, width, height, length]
float32[] mask2d            # 2D Mask for segmented fruit/vegetable 
float32[] mask3d            # 3D Mask for segmented fruit/vegetable 
geometry_msgs/Pose2D pose2d # 2D pose of fruit/vegetable (x, y, theta) 
geometry_msgs/Pose pose3d   # 3D pose of the detected fruit/vegetable
float64 confidence          # Certainty level of the inference
float64 occlusion_level     # Occlusion level to identify how much of the fruit/vegetable is occluded

string pomological_class    # Pomological class of the fruit
                            # Possible values: "Pome" (e.g. Apples, Pears), 
                            # "Drupe" (e.g. Peaches, Cherries), 
                            # "Berry" (e.g. Grapes, Blueberries), 
                            # "Aggregate" (e.g. Raspberries, Strawberries), 
                            # "Multiple" (e.g. Pineapples, Figs), 
                            # "Hesperidium" (e.g. Oranges, Lemons), 
                            # "Pepo" (e.g. Watermelons, Cucumbers), 
                            # "Legume" (e.g. Peas, Beans), "Capsule" (e.g. Poppies, Cotton), 
                            # "Achene" (e.g. Sunflower seeds, Buckwheat), 
                            # "Nut" (e.g. Chestnuts, Hazelnuts), 
                            # "Samara" (e.g. Maples, Ash trees), "Caryopsis" (e.g. Wheat, Rice, Corn), 
                            # "Edible Plant"
                            # "Other",
                            # "Unknown"
                            
string edible_plant_part    # Part of the plant if it is a vegetable instead of a fruit 
                            # Possible values: "Leafy Vegetable" (e.g. Lettuce, Spinach), 
                            # "Culinary Vegetable" (e.g. Tomatoes, Cucumbers, Eggplants)
                            # "Root Vegetable" (e.g. Carrots, Potatoes), 
                            # "Stem Vegetable" (e.g. Asparagus, Leeks), 
                            # "Bulb Vegetable" (e.g. Onions, Garlic), 
                            # "Flower Vegetable" (e.g. Broccoli, Cauliflower), 
                            # "Seed Vegetable" (e.g. Peas, Lentil), "Not Vegetable", "Unknown"
                            # "Other",
                            # "Unknown"

sensor_msgs/Image rgb_image     # RGB image used for detection
sensor_msgs/Image depth_image   # Depth image used for detection