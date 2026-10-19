import json
import os
from contextlib import contextmanager
import yaml
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.engine import HookBase
from detectron2.engine.hooks import BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper, transforms as T
from detectron2.utils.events import EventStorage
from detectron_trainer.detectron_trainer import DetectronTrainer
from detectron_trainer.aoc_trainer import AOCTrainer


ARCHITECTURE = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
CATEGORIES = [
    {"id": 1, "name": "fruit_ripe", "supercategory": ""},
    {"id": 2, "name": "fruit_unripe", "supercategory": ""},
]


class LossGraphHook(HookBase):
    """Measure held-out loss and refresh the graph at each evaluation."""

    def __init__(self, cfg, model, loader):
        self.period = cfg.TEST.EVAL_PERIOD
        self.model = model
        self.loader = loader
        self.output = Path(cfg.OUTPUT_DIR)
        self.history = {"training": [], "validation": []}

    def validation_loss(self):
        # Detectron2 returns loss dictionaries only in training mode. Freeze
        # batch normalization/dropout and restore every module's mode afterward.
        modes = [(module, module.training) for module in self.model.modules()]
        total, count = 0.0, 0
        device = next(self.model.parameters()).device
        devices = [device.index] if device.type == "cuda" else []
        try:
            self.model.train()
            for module in self.model.modules():
                if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm,
                                       torch.nn.modules.dropout._DropoutNd)):
                    module.eval()
            # Keep validation's proposal sampling out of the training RNG and
            # its internal metrics out of the training EventStorage.
            with torch.random.fork_rng(devices=devices), torch.no_grad(), EventStorage():
                for batch in self.loader:
                    losses = self.model(batch)
                    loss = sum(value.item() for value in losses.values())
                    if not np.isfinite(loss):
                        raise ValueError("Non-finite validation loss")
                    total += loss * len(batch)
                    count += len(batch)
        finally:
            for module, training in modes:
                module.training = training
        if not count:
            raise ValueError("Validation dataset is empty")
        return total / count

    def after_step(self):
        iteration = self.trainer.iter + 1
        loss = self.trainer.storage.latest()["total_loss"][0]
        self.history["training"].append([iteration, float(loss)])
        if iteration == self.trainer.max_iter or (self.period > 0 and iteration % self.period == 0):
            validation = self.validation_loss()
            self.history["validation"].append([iteration, validation])
            self.trainer.storage.put_scalar("validation_loss", validation, smoothing_hint=False)
            self.save_graph()
            print(f"Iteration {iteration}: validation loss = {validation:.5f}")

    def save_graph(self):
        self.output.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            training = np.asarray(self.history["training"])
            validation = np.asarray(self.history["validation"])
            ax.plot(training[:, 0], training[:, 1], alpha=0.55,
                    linewidth=0.8, label="Training loss (each iteration)")
            ax.plot(validation[:, 0], validation[:, 1], "o-",
                    label="Validation loss (full validation set)")
            ax.set(xlabel="Training iteration", ylabel="Total loss",
                   title="Strawberry training and validation loss")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            temporary = self.output / "loss_curve.tmp.png"
            fig.savefig(temporary, dpi=150)
            temporary.replace(self.output / "loss_curve.png")
        finally:
            plt.close(fig)
        temporary = self.output / "loss_history.tmp.json"
        temporary.write_text(json.dumps(self.history, indent=2))
        temporary.replace(self.output / "loss_history.json")


def prepare_datasets(dataset, output):
    """Export detector-compatible annotations and metadata without editing inputs."""
    output.mkdir(parents=True, exist_ok=True)
    class_ids = {"ripe": 1, "fruit_ripe": 1,
                 "unripe": 2, "fruit_unripe": 2}
    for split in ("train", "val", "test"):
        image_dir = dataset / split
        source = image_dir / "annotations/ripeness_class_annotations.json"
        annotations = json.loads(source.read_text())
        id_map = {category["id"]: class_ids[category["name"]]
                  for category in annotations["categories"]}
        if set(id_map.values()) != {1, 2}:
            raise ValueError(f"Expected ripe and unripe classes in {source}")
        annotations["categories"] = [dict(category) for category in CATEGORIES]
        images = [image for image in annotations["images"]
                  if (image_dir / image["file_name"]).is_file()]
        print(f"{split}: {len(images)} images; "
              f"omitted {len(annotations['images']) - len(images)} missing images")
        image_ids = {image["id"] for image in images}
        annotations["images"] = images
        annotations["annotations"] = [
            dict(annotation, category_id=id_map[annotation["category_id"]])
            for annotation in annotations["annotations"]
            if annotation["image_id"] in image_ids
        ]
        cleaned = output / f"{split}_annotations.json"
        cleaned.write_text(json.dumps(annotations))


def build_training_hooks(trainer):
    cfg = trainer.cfg
    mapper = DatasetMapper(
        cfg, is_train=True,
        augmentations=[T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)],
    )
    loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)
    return [
        LossGraphHook(cfg, trainer.model, loader),
        BestCheckpointer(cfg.TEST.EVAL_PERIOD, trainer.checkpointer,
                         "segm/AP-fruit_ripe", mode="max", file_prefix="model_best"),
    ]


class StrawberryTrainer(AOCTrainer):
    """Add experiment hooks while inheriting the existing AOC training logic."""

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks[-1:-1] = build_training_hooks(self)
        return hooks


@contextmanager
def training_directory(output):
    # AOCTrainer reads optimizer settings from ./data/config/config.yaml.
    # Supply that layout inside this run's output, without editing shared files.
    previous = Path.cwd()
    os.chdir(output)
    try:
        yield
    finally:
        os.chdir(previous)


def main():
    root = Path(__file__).resolve().parent
    dataset = root / "data/fruit_detector/Datasets/strawberry"
    output = root / "data/training_output/strawberry_ripeness"
    output.mkdir(parents=True, exist_ok=True)

    prepare_datasets(dataset, output)

    # Use DetectronTrainer for configuration, registration, catalogs and evaluation.
    config_path = root.parent / "config/non_ros_params.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["datasets"].update({
        "train_dataset_name": "strawberry_train",
        "validation_dataset_name": "strawberry_val",
        "test_dataset_name": "strawberry_test",
    })
    config["files"].update({
        "config_file": ARCHITECTURE,
        "pretrained_model_file": model_zoo.get_checkpoint_url(ARCHITECTURE),
        # The existing wrapper loads model_file when pretrained_model_file is set.
        "model_file": model_zoo.get_checkpoint_url(ARCHITECTURE),
        "train_annotation_file": str(output / "train_annotations.json"),
        "validation_annotation_file": str(output / "val_annotations.json"),
        "test_annotation_file": str(output / "test_annotations.json"),
        "train_dataset_catalog_file": str(output / "train_dataset_catalog.pkl"),
        "test_metadata_catalog_file": str(output / "test_metadata_catalog.pkl"),
    })
    config["directories"].update({
        "train_image_dir": str(dataset / "train"),
        "validation_image_dir": str(dataset / "val"),
        "test_image_dir": str(dataset / "test"),
        "training_output_dir": str(output),
    })
    config["training"].update({
        "number_of_classes": 2, "epochs": 10000, "learning_rate": 0.00025,
        "num_workers": 2, "batch_size": 2, "steps": [2000, 2600],
        "warmup_iters": 100,
    })
    config["settings"].update({"download_assets": False, "validation_period": 500})
    (output / "training_config.yaml").write_text(yaml.safe_dump(config))
    trainer = DetectronTrainer(config)
    # These run-specific settings belong here; shared trainer classes stay intact.
    cfg = trainer.cfg
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {cfg.MODEL.DEVICE}")
    cfg.OUTPUT_DIR = str(output)
    cfg.DATALOADER.NUM_WORKERS = config["training"]["num_workers"]
    cfg.SOLVER.IMS_PER_BATCH = config["training"]["batch_size"]
    cfg.SOLVER.STEPS = tuple(config["training"]["steps"])
    cfg.SOLVER.WARMUP_ITERS = config["training"]["warmup_iters"]
    optimizer_config = output / "data/config/config.yaml"
    optimizer_config.parent.mkdir(parents=True, exist_ok=True)
    optimizer_config.write_text(yaml.safe_dump(config))
    with training_directory(output):
        aoc_trainer = StrawberryTrainer(cfg)
        aoc_trainer.resume_or_load(resume=False)
        aoc_trainer.train()

    print("Detector configuration (non_ros_params.yaml):")
    best_model = output / "model_best.pth"
    if not best_model.is_file():
        raise RuntimeError("No best model was saved; check validation segm/AP-fruit_ripe metrics")
    print(f"  model_file: '{best_model}'")
    print(f"  test_metadata_catalog_file: '{output / 'test_metadata_catalog.pkl'}'")
    print(f"  config_file: '{ARCHITECTURE}'")

    # Test the validation-selected model, without restoring optimizer state.
    DetectionCheckpointer(aoc_trainer.model).load(str(best_model))
    print("Evaluating model_best.pth on the held-out test set")
    trainer.evaluate_model(aoc_trainer.model)


if __name__ == "__main__":
    main()
