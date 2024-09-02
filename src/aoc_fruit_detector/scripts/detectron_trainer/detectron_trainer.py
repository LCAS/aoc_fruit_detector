"""
Started by: Usman Zahidi (uz) {02/08/24}

"""
import os
#detectron imports
from detectron2.data.datasets import register_coco_instances
from detectron2.config        import get_cfg
from detectron2               import model_zoo
from detectron2.data.catalog  import DatasetCatalog, MetadataCatalog
from detectron2.engine        import DefaultTrainer
from detectron2.evaluation   import COCOEvaluator,inference_on_dataset
from detectron2.data         import build_detection_test_loader

#projects imports
from detectron_trainer.aoc_trainer import AOCTrainer
import pickle, json, logging, traceback
from learner_trainer.learner_trainer import LearnerTrainer
from utils.utils import LearnerUtils

logging.getLogger('detectron2').setLevel(logging.WARNING)

class DetectronTrainer(LearnerTrainer):

    def __init__(self, config_data):
        #UZ:load config data into object variables
        # UZ:dataset
        self.name_train                 = config_data['datasets']['train_dataset_name']
        self.name_test                  = config_data['datasets']['test_dataset_name']
        # UZ:files
        self.model_file                 = config_data['files']['model_file']
        self.config_file                = config_data['files']['config_file']
        self.test_annotation_file       = config_data['files']['test_annotation_file']
        self.train_annotation_file      = config_data['files']['train_annotation_file']
        self.train_dataset_catalog_file = config_data['files']['train_dataset_catalog_file']
        self.test_metadata_catalog_file = config_data['files']['test_metadata_catalog_file']
        self.pretrained_model           = config_data['files']['pretrained_model_file']
        # UZ:training
        self.num_classes                = config_data['training']['number_of_classes']
        self.epochs                     = config_data['training']['epochs']
        self.learning_rate              = config_data['training']['learning_rate']
        # UZ:directories
        self.test_image_dir             = config_data['directories']['test_image_dir']
        self.train_image_dir            = config_data['directories']['train_image_dir']


        downloadUtils=LearnerUtils(config_data)
        downloadUtils.call_download()

        self.cfg = self._configure(self.epochs,self.learning_rate)
        self._register_train_dataset()
        self._register_test_dataset()

    def _configure(self, iterations=10000,
                  learning_rate=0.0025,num_workers=8,batch_size=8,batch_per_image=512,test_threshold=0.5):
        cfg = get_cfg()
        try:
            cfg.merge_from_file(model_zoo.get_config_file(self.config_file))
        except Exception as e:
            logging.error(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)

        # Load pretrained model if path given
        if (self.pretrained_model!=''):
            cfg.MODEL.WEIGHTS = os.path.join(self.model_file)

        #UZ to_do : better read from config file instead of defaults

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.DATASETS.TRAIN = (self.name_train,)
        cfg.DATASETS.TEST = (self.name_test,)
        cfg.DATALOADER.NUM_WORKERS = num_workers
        cfg.SOLVER.IMS_PER_BATCH = batch_size
        cfg.SOLVER.BASE_LR = learning_rate
        cfg.SOLVER.MAX_ITER = iterations
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_per_image
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = test_threshold # set a custom testing threshold
        return cfg

    def _register_test_dataset(self):
        test_dataset_catalog  = MetadataCatalog.get(self.name_test)
        test_dataset_catalog.thing_colors = [(0, 255, 0),(255, 0,0)]
        register_coco_instances(self.name_test, {}, self.test_annotation_file, self.test_image_dir)
        try:
            with open(self.train_annotation_file) as ann_file:
                json_data = json.load(ann_file) #reading categories from annotation file
                categories=json_data['categories']
            pickle.dump([test_dataset_catalog,categories], open(self.test_metadata_catalog_file, 'wb'))
        except Exception as e:
            logging.error(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)

    def _register_train_dataset(self):

        register_coco_instances(self.name_train, {}, self.train_annotation_file, self.train_image_dir)
        train_dataset_catalog = DatasetCatalog.get(self.name_train)

        try:
            pickle.dump(train_dataset_catalog, open(self.train_dataset_catalog_file, 'wb'))
        except Exception as e:
            logging.error(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)

    def train_model(self, resumeType)->DefaultTrainer:
        try:
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
            #UZ: AgriOpenCoreTrainer class doing augmentation
            aoc_trainer = AOCTrainer(self.cfg)
            aoc_trainer.resume_or_load(resume=resumeType)
            aoc_trainer.train()
            return aoc_trainer
        except Exception as e:
            logging.error(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)

    def evaluate_model(self,trainer):
        evaluator = COCOEvaluator(self.name_test, self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(self.cfg, self.name_test)
        print(inference_on_dataset(trainer, val_loader, evaluator))
