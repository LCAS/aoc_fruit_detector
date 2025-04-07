"""
Started by: Usman Zahidi (uz) {01/08/24}
Updated by: Abdurrahman Yilmaz (ayilmaz@lincoln.ac.uk) {07/04/25}
"""
# This file only serves as example to facilitate in integration process
# Training and inference should use the config with parameters
# config contains default values, urls to datasets, catalog files

import os,traceback,cv2,logging,yaml
import numpy as np
from detectron_predictor.json_writer.pycococreator.pycococreatortools.fruit_orientation import FruitTypes
from detectron_trainer.detectron_trainer     import DetectronTrainer
from detectron_predictor.detectron_predictor import DetectronPredictor


from utils.utils import LearnerUtils
from utils.utils import find_path, find_workspace_root

def call_predictor():
    # instantiation
    det_predictor = DetectronPredictor(config_data)

    #loop for generating/saving segmentation output images
    sample_no=1
    depth_file_set = set(depth_files)

    for rgb_file in rgb_files:
        image_file_name=os.path.join(test_image_dir, rgb_file)
        rgb_image   = cv2.imread(image_file_name)

        if rgb_image is None :
            message = 'path to rgb is invalid or inaccessible'
            logging.error(message)
        
        depth_file_candidate = rgb_file.replace(rgb_name_pattern, depth_name_pattern)

        depth_path = os.path.join(test_image_dir, depth_file_candidate)
        depth_image = None
        
        if depth_file_candidate in depth_file_set and os.path.exists(depth_path):
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) 
            if depth_image is None:
                logging.warning(f"Depth file {depth_path} could not be read, falling back to blue channel.")
        if depth_image is not None:
            # Resize depth to match rgb if needed
            if (depth_image.shape[0] != rgb_image.shape[0]) or (depth_image.shape[1] != rgb_image.shape[1]):
                depth_image = cv2.resize(depth_image, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            if len(depth_image.shape) == 2:  # depth is single channel
                depth_image = np.expand_dims(depth_image, axis=2)
            elif len(depth_image.shape) == 3 and depth_image.shape[2] == 3:
                depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
                depth_image = np.expand_dims(depth_image, axis=2)
            
            rgbd_image = np.concatenate((rgb_image, depth_image), axis=2)
        else:
            rgbd_image = np.dstack([rgb_image,rgb_image[:,:,0]]) #making blue as depth

        # ** main call **
        try:
            filename,extension = os.path.splitext(rgb_file)
            if (prediction_json_output_dir!=""):
                prediction_json_output_file = os.path.join(prediction_json_output_dir, filename)+'.json'
            else:
                prediction_json_output_file = ""

            if (__debug__):

                json_annotation_message, predicted_image, depth_masks = det_predictor.get_predictions_image(rgbd_image,
                                                                                                            prediction_json_output_file,
                                                                                                            prediction_image_output_dir,
                                                                                                            image_file_name,
                                                                                                            sample_no,fruit_type)
            else:
                dummy_image_id=1
                json_annotation_message, predicted_image, rgb_masks, depth_masks = det_predictor.get_predictions_message(rgbd_image, dummy_image_id,fruit_type)
            # Use output json_annotation_message,predicted_image as per requirement
            # In Optimized (non-debug) mode predicted_image is None

            yield json_annotation_message

        except Exception as e:
            logging.error(e)
            print(traceback.format_exc()) if __debug__ else print(e)
        sample_no += 1
    call_trainer(False,True) # call for evaluation

def call_trainer(resumeType=False,skipTraining=False):

    try:
        detTrainer=DetectronTrainer(config_data)
        aoc_trainer=detTrainer.train_model(resumeType=resumeType,skipTraining=skipTraining) # set resumeType=True when continuing training on top of parly trained models
        detTrainer.evaluate_model(aoc_trainer.model)
    except Exception as e:
        logging.error(e)
        print(traceback.format_exc()) if __debug__ else print(e)

workspace_dir = find_workspace_root('fruit_detector_ws')
search_root = os.path.join(workspace_dir, 'src', 'aoc_fruit_detector') # 'src' or 'install' based on user desire
config_dir = find_path('config', search_root=search_root, search_type='dir')


if not config_dir:
    raise FileNotFoundError(f"Could not find 'config' directory under {search_root}")

config_name = 'non_ros_params.yaml'
config_path = find_path('non_ros_params.yaml', search_root=config_dir, search_type='file')


if config_path:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
else:
    raise FileNotFoundError(f"No '{config_name}' found in '{config_dir}' or any of its subdirectories")

for section in ['files', 'directories']:
    if section in config_data:
        for key, path in config_data[section].items():
            if isinstance(path, str) and path.startswith('./'):
                config_data[section][key] = find_path(path.lstrip('./'), search_root=search_root, search_type='any')


name_train                  = config_data['datasets']['train_dataset_name']
name_test                   = config_data['datasets']['test_dataset_name']

train_image_dir             = config_data['directories']['train_image_dir']
test_image_dir              = config_data['directories']['test_image_dir']
prediction_json_output_dir  = config_data['directories']['prediction_json_dir']
prediction_image_output_dir = config_data['directories']['prediction_output_dir']

num_classes                 = config_data['training']['number_of_classes']
epochs                      = config_data['training']['epochs']
download_assets             = config_data['settings']['download_assets']
fruit_type                  = config_data['settings']['fruit_type']
rgb_name_pattern            = config_data['settings']['filename_patterns']['rgb']
depth_name_pattern          = config_data['settings']['filename_patterns']['depth']

# UZ: utils call is made here because we are looping through image directory which is empty in the beginning.
# This call might be unnecessary in other use cases

if (download_assets):
    downloadUtils=LearnerUtils(config_data)
    downloadUtils.call_download()

if (fruit_type.upper()=="STRAWBERRY"):
    fruit_type=FruitTypes.Strawberry
elif (fruit_type.upper()=="TOMATO"):
    fruit_type=FruitTypes.Tomato
else:
    fruit_type=FruitTypes.Strawberry

test_image_dir = os.path.normpath(test_image_dir)

# List files
all_files = sorted([
    f for f in os.listdir(test_image_dir)
    if os.path.isfile(os.path.join(test_image_dir, f))
])

# Separate them into RGB and depth lists
rgb_files = sorted([
    f for f in all_files
    if rgb_name_pattern in f
])

depth_files = sorted([
    f for f in all_files
    if depth_name_pattern in f
])

print(f"Found {len(rgb_files)} RGB files in '{test_image_dir}'")
print(f"Found {len(depth_files)} depth files in '{test_image_dir}'")

if __name__ == '__main__':
    for prediction in call_predictor():
        print("Prediction generated.")


