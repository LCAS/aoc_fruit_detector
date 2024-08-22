# Lincoln Institue of Agri-food Technology (LIAT)
# Contributors:
# Usman Zahidi (Started: 04/08/2024)

import os
import json
import logging
import numpy as np
import matplotlib.figure as mplfigure
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from .pycococreator.pycococreatortools import pycococreatortools

#detectron imports
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer,GenericMask,ColorMode


from .utils.colormap import random_color
import datetime
logger = logging.getLogger(__name__)
__all__ = ["ColorMode", "VisImage", "Visualizer"]

class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img                = img
        self.scale              = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)



    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax


    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)


    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

class JSONWriter(Visualizer):
    """
        writes visualization of predictions from a model into COCO 1.0 json annotations format
    """

    annotation_id=1
    def __init__(self,img_rgb,metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super(JSONWriter, self).__init__(img_rgb, metadata, scale, instance_mode)
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): image metadata.
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        self.scale=scale
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        if metadata is None:
            metadata = MetadataCatalog.get("__nonexist__")
        self.metadata = metadata
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        self._instance_mode = instance_mode

    def create_prediction_json(self, predictions, output_json_file_path,input_file_names,categories_info):

        image_list = list()
        category_list = list()
        dict_info=pycococreatortools.create_info()
        dict_license=pycococreatortools.create_license_info()
        dict_category={"categories":categories_info}
        image_id=1

        # UZ: if input filename is string (only one name) then make it list

        if type(input_file_names) is str:
            str_holder=input_file_names
            input_file_names=list()
            input_file_names.append(str_holder)

        for input_file_name in input_file_names:
            head,filename = os.path.split(input_file_name)
            image_list.append(pycococreatortools.create_image_info(image_id,filename,
            [self.output.height,self.output.width],datetime.datetime.utcnow().isoformat(' ')))
            ann_list=self._convert_instance_predictions_to_annotations(predictions, input_file_name,
                                                           output_json_file_path,image_id)
            image_id += 1
        dict_images={"images": image_list}
        dict_annotations={"annotations": ann_list}
        
        json_output_dict = {**dict_info, **dict_license, **dict_images,**dict_annotations,**dict_category}
        # UZ: call self._write_to_file for dumping to file
        if(__debug__):
            self._write_to_file(output_json_file_path, json_output_dict)
        return json_output_dict

    def _write_to_file(self,output_json_file_path,json_output_dict):

        json_dir, local_filename = os.path.split(output_json_file_path)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        with open(output_json_file_path, 'w') as output_json_file:
            json.dump(json_output_dict,output_json_file,skipkeys=False, ensure_ascii=True,
                      check_circular=True, allow_nan=True, cls=None, indent=2, separators=None,
                      default=None, sort_keys=False)


    def _convert_instance_predictions_to_annotations(self, predictions,image_filename="",
                                             json_filename="",image_id=0)->None:

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels      =None
        keypoints   =None
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        return self._overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            image_filename=image_filename,
            image_id = image_id,
            classes = classes,
        )

    def _overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        image_filename="",
        image_id=0,
        classes=None,
    ):
        ann_list= list()
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)

        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output

        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]


        if masks is not None:
            index=0
            for segment in masks:
                image = Image.open(image_filename)
                category_info = {'id': int(classes[index]+1), 'is_crowd': '0'}
                annotation_info = pycococreatortools.create_annotation_info(
                    self.annotation_id, image_id, category_info, segment.mask, classes[index]+1,
                    image.size, True, 0)
                ann_list.append(annotation_info)
                self.annotation_id +=1
                index+=1

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                bbox=list(boxes[i]);

            if labels is not None:
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < super._SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )
        return ann_list
