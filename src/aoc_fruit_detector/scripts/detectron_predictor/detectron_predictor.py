"""
Started by: Usman Zahidi (uz) {02/08/24}
"""
# general imports
import os, pickle, logging,traceback

# detectron imports
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2 import model_zoo

# project imports
from detectron_predictor.visualizer.aoc_visualizer import AOCVisualizer, ColorMode
from detectron_predictor.json_writer.JSONWriter import JSONWriter
from learner_predictor.learner_predictor import LearnerPredictor
from utils.utils import LearnerUtils
import cv2


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
        except Exception as e:
            logging.error(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)
        return data

    def get_predictions(self, rgb_image,output_json_file_path,image_file_name):
        predicted_image=None
        try:
            outputs = self.predictor(rgb_image)
            vis_aoc = AOCVisualizer(rgb_image,
                                   metadata=self.metadata[0],
                                   scale=self.scale,
                                   instance_mode=self.instance_mode
                                   )
            predictions = outputs["instances"].to("cpu")
            if (__debug__):
                drawn_predictions = vis_aoc.draw_instance_predictions(outputs["instances"].to("cpu"))
                predicted_image = drawn_predictions.get_image()[:, :, ::-1].copy()
                pred_image_dir = os.path.join(self.cfg.OUTPUT_DIR, 'predicted_images')
                if not os.path.exists(pred_image_dir):
                    os.makedirs(pred_image_dir)
                file_dir,f_name = os.path.split(image_file_name)
                overlay_fName = os.path.join(pred_image_dir, f_name)
                cv2.imwrite(overlay_fName, cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))
                print(f"predicted image saved in output folder for file {overlay_fName}")
            json_writer = JSONWriter(rgb_image, self.metadata[0])
            categories_info=self.metadata[1] # category info is saved as second list
            predicted_json_ann=json_writer.create_prediction_json(predictions, output_json_file_path, image_file_name,categories_info)
            return predicted_json_ann,predicted_image
        except Exception as e:
            logging.error(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)


