"""
Started by: Usman Zahidi (uz) {02/08/24}
"""
# general imports
import os, pickle, logging,traceback

import numpy
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2 import model_zoo
from detectron_predictor.json_writer.pycococreator.pycococreatortools.fruit_orientation import FruitTypes

# project imports
from detectron_predictor.visualizer.aoc_visualizer import AOCVisualizer, ColorMode
from detectron_predictor.json_writer.JSONWriter import JSONWriter
from learner_predictor.learner_predictor import LearnerPredictor
from utils.utils import LearnerUtils
import cv2
import numpy as np
from datetime import datetime


logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


class DetectronPredictor(LearnerPredictor):

    def __init__(self, config_data, scale=1.0,
                 instance_mode=ColorMode.SEGMENTATION):

        self.predictor = None
        self.instance_mode = instance_mode
        self.scale = scale

        self.model_file             = config_data['files']['model_file']
        self.config_file            = config_data['files']['config_file']
        self.metadata_file          = config_data['files']['test_metadata_catalog_file']
        self.dataset_catalog_file   = config_data['files']['train_dataset_catalog_file']
        self.num_classes            = config_data['training']['number_of_classes']
        self.epochs                 = config_data['training']['epochs']
        self.download_assets        = config_data['settings']['download_assets']
        self.rename_pred_images     = config_data['settings']['rename_pred_images']
        self.show_orientation       = config_data['settings']['show_orientation']
        self.bbox                   = config_data['settings']['bbox']
        self.masks                  = config_data['settings']['segm_masks']
        self.list_category_ids      = list()
        self.colours                = None
        self.CONFIDENCE_THRESHOLD   = config_data['settings']['confidence_threshold']
        self.CONFIDENCE_THRESHOLD   = max(0, min(self.CONFIDENCE_THRESHOLD, 1)) # ensure the value is between 0 and 1


        if (self.download_assets):
            downloadUtils=LearnerUtils(config_data)
            downloadUtils.call_download()

        self.metadata = self._get_catalog(self.metadata_file)
        self.cfg = self._configure()


        try:
            self.predictor = DefaultPredictor(self.cfg)
        except Exception as e:
            logging.error(e)
            print(traceback.format_exc())
            raise Exception(e)


    def _configure(self):
        cfg = get_cfg()

        try:
            cfg.merge_from_file(model_zoo.get_config_file(self.config_file))
        except Exception as e:
            logging.error(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)

        cfg.MODEL.WEIGHTS = os.path.join(self.model_file)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        return cfg


    def _get_catalog(self,catalog_file):

        # metadata file has name of classes, it is created to avoid having custom dataset and taking definitions
        # from annotations, instead. It has structure of MetaDataCatlog output of detectron2
        try:
            file = open(catalog_file, 'rb')
            data = pickle.load(file)
            file.close()

            self.colours=numpy.asarray(data[0].as_dict()['thing_colors'][::-1])
            categories=data[-1]
            for category in categories:
                self.list_category_ids.append(category['id']-1)

        except Exception as e:
            logging.error(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)
        return data
    
    def get_rgb_predictions_image(self, rgb_image, output_json_file_path='',prediction_output_dir='',image_file_name='',sample_no=1,fruit_type=FruitTypes.Strawberry):
        rgb_image = rgb_image[:, :, :3].astype(np.uint8)
        image_size = rgb_image.shape
        image_size = tuple(reversed(image_size[:-1]))
        save_json_file = True
        try:
            outputs = self.predictor(rgb_image)
            predictions = outputs["instances"].to("cpu")

            scores = predictions.scores
            keep = scores > self.CONFIDENCE_THRESHOLD
            indices = keep.nonzero(as_tuple=True)[0]  # Get valid indices
            predictions = predictions[indices]

            vis_aoc = AOCVisualizer(rgb_image,
                                    metadata=self.metadata[0],
                                    scale=self.scale,
                                    instance_mode=self.instance_mode,
                                    colours=self.colours,
                                    category_ids=self.list_category_ids,
                                    masks=self.masks,
                                    bbox=self.bbox,
                                    show_orientation=self.show_orientation,
                                    fruit_type=fruit_type
                                    )
            start_time = datetime.now()
            drawn_predictions = vis_aoc.draw_instance_predictions(predictions)
            end_time = datetime.now()
            predicted_image = drawn_predictions.get_image()[:, :, ::-1].copy()            
            
            pred_image_dir = os.path.join(prediction_output_dir, 'predicted_images')
            if not os.path.exists(pred_image_dir):
                os.makedirs(pred_image_dir)

            if (self.rename_pred_images):
                f_name=f'img_{str(sample_no).zfill(6)}.png'
                overlay_fName = os.path.join(pred_image_dir, f_name)
                file_dir, f = os.path.split(output_json_file_path)
                image_file_name = f_name
                f_name = f'img_{str(sample_no).zfill(6)}.json'
                output_json_file_path = os.path.join(file_dir, f_name)

            else:
                file_dir, f_name = os.path.split(image_file_name)
                overlay_fName = os.path.join(pred_image_dir, f_name)
            cv2.imwrite(overlay_fName, cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))
            delta=str(end_time - start_time)
            print(f"Predicted image saved in output folder for file {overlay_fName}, Duration: {delta}")
            json_writer = JSONWriter(rgb_image, self.metadata[0])
            categories_info=self.metadata[1] # category info is saved as second list
            predicted_json_ann=json_writer.create_prediction_json(predictions, output_json_file_path, image_file_name,categories_info,image_size,1,save_json_file)
            return predicted_json_ann,predicted_image,[]
        except Exception as e:
            logging.error(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)

    def get_predictions_image(self, rgbd_image,output_json_file_path='',prediction_output_dir='',image_file_name='',sample_no=1,fruit_type=FruitTypes.Strawberry):

        depth_image = rgbd_image[:, :, 3]
        rgb_image = rgbd_image[:, :, :3].astype(np.uint8)
        image_size = rgb_image.shape
        image_size = tuple(reversed(image_size[:-1]))
        save_json_file = True
        try:
            outputs = self.predictor(rgb_image)
            predictions = outputs["instances"].to("cpu")

            scores = predictions.scores
            keep = scores > self.CONFIDENCE_THRESHOLD
            indices = keep.nonzero(as_tuple=True)[0]  # Get valid indices
            predictions = predictions[indices]

            vis_aoc = AOCVisualizer(rgb_image,
                                    metadata=self.metadata[0],
                                    scale=self.scale,
                                    instance_mode=self.instance_mode,
                                    colours=self.colours,
                                    category_ids=self.list_category_ids,
                                    masks=self.masks,
                                    bbox=self.bbox,
                                    show_orientation=self.show_orientation,
                                    fruit_type=fruit_type
                                    )
            start_time = datetime.now()
            drawn_predictions = vis_aoc.draw_instance_predictions(predictions)
            end_time = datetime.now()
            predicted_image = drawn_predictions.get_image()[:, :, ::-1].copy()

            depth_masks, rgb_masks = self.get_masks(predicted_image, rgb_image, depth_image)
            
            pred_image_dir = os.path.join(prediction_output_dir, 'predicted_images')
            if not os.path.exists(pred_image_dir):
                os.makedirs(pred_image_dir)

            if (self.rename_pred_images):
                f_name=f'img_{str(sample_no).zfill(6)}.png'
                overlay_fName = os.path.join(pred_image_dir, f_name)
                file_dir, f = os.path.split(output_json_file_path)
                image_file_name = f_name
                f_name = f'img_{str(sample_no).zfill(6)}.json'
                output_json_file_path = os.path.join(file_dir, f_name)

            else:
                file_dir, f_name = os.path.split(image_file_name)
                overlay_fName = os.path.join(pred_image_dir, f_name)
            cv2.imwrite(overlay_fName, cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))
            delta=str(end_time - start_time)
            print(f"Predicted image saved in output folder for file {overlay_fName}, Duration: {delta}")
            json_writer = JSONWriter(rgb_image, self.metadata[0])
            categories_info=self.metadata[1] # category info is saved as second list
            predicted_json_ann=json_writer.create_prediction_json(predictions, output_json_file_path, image_file_name,categories_info,image_size,1,save_json_file)
            return predicted_json_ann,predicted_image,depth_masks
        except Exception as e:
            logging.error(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)

    def get_predictions_message(self, rgbd_image, image_id=0,fruit_type=FruitTypes.Strawberry):
        depth_image = rgbd_image[:, :, 3]
        rgb_image = rgbd_image[:, :, :3].astype(np.uint8)
        output_json_file_path=''
        image_size=rgb_image.shape
        image_size = tuple(reversed(image_size[:-1]))
        image_file_name= f'img_{str(image_id).zfill(6)}.png'
        save_json_file=False
        try:
            outputs = self.predictor(rgb_image)
            predictions = outputs["instances"].to("cpu")

            scores = predictions.scores
            keep = scores > self.CONFIDENCE_THRESHOLD
            indices = keep.nonzero(as_tuple=True)[0]  # Get valid indices
            predictions = predictions[indices]

            vis_aoc = AOCVisualizer(rgb_image,
                                    metadata=self.metadata[0],
                                    scale=self.scale,
                                    instance_mode=self.instance_mode,
                                    colours=self.colours,
                                    category_ids=self.list_category_ids,
                                    masks=self.masks,
                                    show_orientation=False,   # UZ: Set to false
                                    fruit_type=fruit_type,
                                    )
            drawn_predictions = vis_aoc.draw_instance_predictions(predictions)
            predicted_image = drawn_predictions.get_image()[:, :, ::-1].copy()
            depth_masks, rgb_masks = self.get_masks(predicted_image, rgb_image, depth_image)
            json_writer = JSONWriter(rgb_image, self.metadata[0],fruit_type)
            categories_info = self.metadata[1]  # category info is saved as second list
            predicted_json_ann = json_writer.create_prediction_json(predictions, output_json_file_path,
                                                                    image_file_name, categories_info, image_size, image_id, save_json_file)
            return predicted_json_ann, predicted_image, rgb_masks, depth_masks
        except Exception as e:
            logging.error(e)
            raise Exception(e)
    
    def get_predictions_message_short(self, rgbd_image, image_id=0,fruit_type=FruitTypes.Strawberry):
        rgb_image = rgbd_image[:, :, :3].astype(np.uint8)
        output_json_file_path=''
        image_size=rgb_image.shape
        image_size = tuple(reversed(image_size[:-1]))
        image_file_name= f'img_{str(image_id).zfill(6)}.png'
        save_json_file=False
        try:
            outputs = self.predictor(rgb_image)
            predictions = outputs["instances"].to("cpu")
            
            scores = predictions.scores
            keep = scores > self.CONFIDENCE_THRESHOLD
            indices = keep.nonzero(as_tuple=True)[0]  # Get valid indices
            predictions = predictions[indices]

            json_writer = JSONWriter(rgb_image, self.metadata[0], fruit_type)
            categories_info = self.metadata[1]  # category info is saved as second list
            predicted_json_ann = json_writer.create_prediction_json(predictions, output_json_file_path,
                                                                    image_file_name, categories_info, image_size, image_id, save_json_file)
            return predicted_json_ann
        except Exception as e:
            logging.error(e)
            raise Exception(e)

    def get_masks(self, fg_masks, rgb_image, depth_image):

        # input three foreground class' masks and calculate leftover as background mask
        # then output requested depth masks as per class_list order

        first_iter=True
        for colour, category_id in zip(self.colours, self.list_category_ids):
            class_colour = np.bitwise_and(fg_masks[:, :, 1] == colour[1], fg_masks[:, :, 0] == colour[0])
            class_colour = np.bitwise_and(class_colour == True, fg_masks[:, :, 2] == colour[2])*1
            depth_mask = class_colour*depth_image
            class_colour *= (category_id+1)

            if first_iter:
                classwise_segm_masks = class_colour
                classwise_depth_masks = depth_mask
                first_iter = False
            else:
                classwise_segm_masks  = np.dstack([classwise_segm_masks,class_colour])
                classwise_depth_masks = np.dstack([classwise_depth_masks,depth_mask])
        return classwise_depth_masks, classwise_segm_masks


