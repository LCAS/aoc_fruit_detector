import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask


convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='A'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))
    return rle

def binary_mask_to_polygon(binary_mask, tolerance=2):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask,0.5)

    cont_iter = iter(contours)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        next(cont_iter)
        polygons.append(segmentation)

    return polygons

def create_image_info(image_id, file_name, image_size, 
                      date_captured="",
                      license_id=0, coco_url="", flickr_url="n/a",tag_ids=[]):

    image_info = {
            "license": license_id,
            "file_name": file_name,
            "coco_url": coco_url,
            "height": image_size[0],
            "width": image_size[1],
            "date_captured": "",
            "flickr_url": flickr_url,
            "darwin_url": "",
            "darwin_workview_url":"",
            "id": image_id
            #"tag_ids" : tag_ids
    }

    return image_info

def create_info(description="Exported from Darwin", url="https://www.lincoln.ac.uk/home/liat/", version="1.0", year=2021,
                           contributor="Lincoln Institute of Agri-food Technology", date_created=datetime.datetime.utcnow().isoformat(' ')):

    info = {"info":{
            "description": description,
            "url": url,
            "version": version,
            "year": year,
            "contributor": contributor,
            "date_created": date_created
    }}

    return info


def create_license_info(url="https://www.lincoln.ac.uk/home/liat/", license_id="1", name="placeholder license"):

    license_info = {"licenses":
            [{"url": url,
            "id": license_id,
            "name": name}]
    }

    return license_info

def create_categories_info(category_id=1, name="none", supercategory="root"):

    category_info = {
            "id": category_id,
            "name": name,
            "supercategory": supercategory}

    return category_info


def create_annotation_info(annotation_id, image_id, category_info, binary_mask,class_id, image_size, include_keypoints,tolerance, bounding_box=None):

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    is_crowd = 0
    segmentation = binary_mask_to_polygon(binary_mask, tolerance)
    if len(segmentation)>=1:
        seg=segmentation[0]
    else:
        seg = segmentation

    annotation_info = {"id": annotation_id,
    "image_id": image_id,
    "category_id": category_info["id"],
    "segmentation": segmentation,
    "area": area.tolist(),
    "bbox": bounding_box.tolist(),
    "iscrowd": is_crowd,
    "num_keypoints":1}

    return annotation_info
