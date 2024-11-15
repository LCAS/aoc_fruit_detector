"""
Started by: Usman Zahidi (uz) {01/08/24}
Updated by: Abdurrahman Yilmaz (ayilmaz@lincoln.ac.uk) {06/09/24}
"""
# This file only serves as example to facilitate in integration process
# Training and inference should use the config with parameters
# config contains default values, urls to datasets, catalog files

import os,traceback,cv2,logging,yaml
import numpy as np

from detectron_trainer.detectron_trainer     import DetectronTrainer
from detectron_predictor.detectron_predictor import DetectronPredictor


from utils.utils import LearnerUtils

def find_data_folder_config(search_dir='.'):
    for root, dirs, _ in os.walk(search_dir):
        if 'data' in dirs:
            data_folder = os.path.join(root, 'data')
            config_path = os.path.join(data_folder, 'config', 'config.yaml')
            # Check if config.yaml exists in the data/config folder
            if os.path.exists(config_path):
                return config_path
    return None

def find_config_file(config_name='parameters.yaml', search_dir='.'):
    for root, dirs, files in os.walk(search_dir):
        if config_name in files:
            return os.path.join(root, config_name)
    return None

#config_path = find_data_folder_config()
config_name = 'parameters.yaml'
search_dir = 'aoc_fruit_detector/config'
config_path = find_config_file(config_name, search_dir)
ref_mask_file_path=os.path.join(search_dir,'reference_mask','straw_reference_mask.png')

if config_path:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
else:
    #raise FileNotFoundError(f"No config file found in any 'data/config/' folder within {os.getcwd()}")
    raise FileNotFoundError(f"No '{config_name}' found in '{search_dir}' or any of its subdirectories")

name_train                  = config_data['datasets']['train_dataset_name']
name_test                   = config_data['datasets']['test_dataset_name']

train_image_dir             = config_data['directories']['train_image_dir']
test_image_dir              = config_data['directories']['test_image_dir']
prediction_json_output_dir  = config_data['directories']['prediction_json_dir']
prediction_image_output_dir = config_data['directories']['prediction_output_dir']

num_classes                 = config_data['training']['number_of_classes']
epochs                      = config_data['training']['epochs']
download_assets             = config_data['settings']['download_assets']

# UZ: utils call is made here because we are looping through image directory which is empty in the beginning.
# This call might be unnecessary in other use cases

if (download_assets):
    downloadUtils=LearnerUtils(config_data)
    downloadUtils.call_download()

rgb_files = sorted([f for f in os.listdir(test_image_dir) if os.path.isfile(os.path.join(test_image_dir,f))])


def call_predictor()->None:

    # instantiation
    det_predictor = DetectronPredictor(config_data)

    #loop for generating/saving segmentation output images
    sample_no=1
    for rgb_file in rgb_files:
        image_file_name=os.path.join(test_image_dir, rgb_file)
        ref_mask= cv2.imread(ref_mask_file_path)
        rgb_image   = cv2.imread(image_file_name)
        rgbd_image = np.dstack([rgb_image,rgb_image[:,:,0]]) #making blue as depth

        if rgb_image is None :
            message = 'path to rgb is invalid or inaccessible'
            logging.error(message)

        if ref_mask is None :
            message = 'path to reference mask is invalid or inaccessible'
            logging.error(message)

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
                                                                                                            image_file_name,
                                                                                                            sample_no,
                                                                                                            ref_mask)

            else:
                dummy_image_id=1
                json_annotation_message, predicted_image,depth_masks = det_predictor.get_predictions_message(rgbd_image, dummy_image_id,ref_mask)
            # Use output json_annotation_message,predicted_image as per requirement
            # In Optimized (non-debug) mode predicted_image is None

            yield json_annotation_message

        except Exception as e:
            logging.error(e)
            print(traceback.format_exc()) if __debug__ else print(e)
        sample_no += 1
    call_trainer(False,True) # call for evaluation
def call_trainer(resumeType=False,skipTraining=False)->None:

    try:
        detTrainer=DetectronTrainer(config_data)
        aoc_trainer=detTrainer.train_model(resumeType=resumeType,skipTraining=skipTraining) # set resumeType=True when continuing training on top of parly trained models
        detTrainer.evaluate_model(aoc_trainer.model)
    except Exception as e:
        logging.error(e)
        print(traceback.format_exc()) if __debug__ else print(e)




