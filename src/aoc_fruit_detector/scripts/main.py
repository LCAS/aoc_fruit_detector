"""
Started by: Usman Zahidi (uz) {01/08/24}
"""
# This file only serves as example to facilitate in integration process
# Training and inference should use the config with parameters
# config contains default values, urls to datasets, catalog files

import os,traceback,cv2,logging,yaml
from detectron_trainer.detectron_trainer     import DetectronTrainer
from detectron_predictor.detectron_predictor import DetectronPredictor
from os import listdir

from utils.utils             import LearnerUtils

with open('./data/config/config.yaml', 'r') as file:
   config_data = yaml.safe_load(file)

name_train                  = config_data['datasets']['train_dataset_name']
name_test                   = config_data['datasets']['test_dataset_name']

train_image_dir             = config_data['directories']['train_image_dir']
test_image_dir              = config_data['directories']['train_image_dir']
prediction_json_output_dir  = config_data['directories']['prediction_json_dir']
prediction_image_output_dir = config_data['directories']['prediction_output_dir']

num_classes                 = config_data['training']['number_of_classes']
epochs                      = config_data['training']['epochs']

# UZ: utils call is made here because we are looping through image directory which is empty in the beginning.
# This call might be unnecessary in other use cases

downloadUtils=LearnerUtils(config_data)
downloadUtils.call_download()

rgb_files=listdir(test_image_dir)

def call_predictor()->None:

    # instantiation
    det_predictor = DetectronPredictor(config_data)

    #loop for generating/saving segmentation output images
    for rgb_file in rgb_files:
        image_file_name=os.path.join(test_image_dir, rgb_file)
        rgb_image   = cv2.imread(image_file_name)

        if rgb_image is None :
            message = 'path to rgb is invalid or inaccessible'
            logging.error(message)

        # ** main call **
        try:
            filename,extension = os.path.splitext(rgb_file)
            if (prediction_json_output_dir!=""):
                prediction_json_output_file = os.path.join(prediction_json_output_dir, filename)+'.json'
            else:
                prediction_json_output_file = ""
            json_annotation_message,predicted_image = det_predictor.get_predictions(rgb_image, prediction_json_output_file,
                                                                                image_file_name)
            # Use output json_annotation_message,predicted_image as per requirement
            # In Optimized (non-debug) mode predicted_image is None
        except Exception as e:
            logging.error(e)
            print(traceback.format_exc()) if __debug__ else print(e)

def call_trainer()->None:

    try:
        detTrainer=DetectronTrainer(config_data)
        aoc_trainer=detTrainer.train_model(resumeType=False) # set resumeType=True when continuing training on top of parly trained models
        detTrainer.evaluate_model(aoc_trainer.model);
    except Exception as e:
        logging.error(e)
        print(traceback.format_exc()) if __debug__ else print(e)

if __name__ == '__main__':
    #Call the trainer or predictor
    #call_trainer()
    call_predictor()