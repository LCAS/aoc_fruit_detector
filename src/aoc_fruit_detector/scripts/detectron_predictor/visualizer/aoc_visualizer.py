"""
Started by: Usman Zahidi (uz) {16/02/22}

"""

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplc

# detectron2 imports
from detectron2.utils.visualizer import Visualizer,ColorMode,VisImage

_SMALL_OBJECT_AREA_THRESH = 1000

class AOCVisualizer(Visualizer):

    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.SEGMENTATION):
        super(AOCVisualizer,self).__init__(img_rgb, metadata, scale, instance_mode)


    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5
    )->VisImage:
        """
        uz: overrided function to customize masks as per AOC requirements i.e. json

        Returns:
            output (VisImage): image object with visualizations.
        """

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
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [[1.0,0.0,0.0],[0.0,1.0,0.0]]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.

            org_labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

            # uz: override colors, remove boxes and labels
            assigned_colors = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0],[1.0, 0.0, 0.0]]
            #boxes           = None
            labels          = None

        for i in range(num_instances):

            #uz: assign colors according to label text from metadata

            if "0 " in org_labels[i]:
                color = assigned_colors[0]
            elif "1 " in org_labels[i]:
                color = assigned_colors[1] #assigned_colors[2]
            #elif ClassNames.STRAWBERRY.name.lower() in org_labels[i]:
            #    color = assigned_colors[2]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
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
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
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
                    org_labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5)->VisImage:
        """
        uz: overrided function to change alpha and remove outline colouring unnecessary code

        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        # uz: change alpha to 1 for creating masks
        alpha = 0.1
        edge_color = color
        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output


