"""
Started by: Usman Zahidi (uz) {02/08/24}
"""
import cv2
import torch
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.projects.deeplab import  build_lr_scheduler
from detectron2.data import detection_utils as utils
from detectron2.config import CfgNode
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
import copy,yaml

#UZ: extended Default trainer to have methods for augmentation

class AOCTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_train_augmentation(self,cfg):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        augs.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
        augs.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
        augs.append(T.RandomContrast(0.8, 3))
        augs.append(T.RandomBrightness(0.3, 1.6))
        return augs

    #UZ: overriding the function to change optimizer from config file. No easy way otherwise in detectron2
    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )
        try:
            with open('./data/config/config.yaml', 'r') as file:
                config_data = yaml.safe_load(file)
            optimizer = config_data['training']['optimizer']
            if (optimizer.upper()=='ADAM'):
                return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
                    params,
                    lr=cfg.SOLVER.BASE_LR,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                )
            else:
                return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                    params,
                    lr=cfg.SOLVER.BASE_LR,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                )
        except Exception as e: #UZ: if invalid read from config then default to SGD

            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=cls.build_train_augmentation(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def custom_mapper(cls,dataset_dict):
        # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
        dataset_dicts = copy.deepcopy(dataset_dict)  # it will be modified by code below
        for d in dataset_dicts:
            image = utils.read_image(d["file_name"], format="BGR")
            transform_list = [T.Resize((400,400)),
                T.RandomBrightness(0.3, 1.5),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            ]
            image, transforms = T.apply_transform_gens(transform_list, image)
            d["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in d.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            d["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

    #UZ: Utility function for trainer for YUV conv from custom dataset
    @classmethod
    def hsv_convert(cls,dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        for d in dataset_dict:
            image = cv2.imread(d["file_name"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            d["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        return dataset_dict
