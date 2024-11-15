"""
Started by: Usman Zahidi (uzahidi@lincoln.ac.uk) {08/11/24}

"""
from reportlab.graphics.widgets.grids import centroid
from skimage.transform import warp_polar
from skimage.registration import phase_cross_correlation
import numpy as np
import cv2
from math import atan2
from enum import Enum,unique
import matplotlib.pyplot as plt

RADIUS = 300

@unique
class OrientationMethod(Enum):
    """
    Enum of different class names
    """
    """
    Principal Component Analysis
    """
    PCA         = 0
    """
    Log Polar Transform (Correlation of phase of mask in frequency domain to estimate angle with respect to reference image)
    """
    LOG_POLAR   = 1


class FruitOrientation():

    @staticmethod
    def get_angle_logpolar(mask,ref_mask):
        # UZ: Log-polar transform requires a reference mask and the orientation is with respect to it
        try:
            if (mask is not None):
                y, x = np.where(mask)
                xy = np.column_stack([x, y])
                centroid = xy.mean(axis=0)
                centroid=[int(centroid[0]),int(centroid[1])]
                ref_mask = ref_mask.astype('float')
                new_mask = mask.astype('float')
                new_mask = cv2.resize(new_mask, dsize=(ref_mask.shape[1], ref_mask.shape[0]), interpolation=cv2.INTER_CUBIC)
                new_mask = np.dstack((new_mask, new_mask, new_mask))
                image_polar = warp_polar(ref_mask, radius=RADIUS, channel_axis=-1)
                rotated_polar = warp_polar(new_mask, radius=RADIUS, channel_axis=-1)
                shifts, _, _ = phase_cross_correlation(
                    rotated_polar,image_polar, normalization=None
                )
                #phase_cross_correlation calculates degrees
                return round(shifts[0],3),centroid
            else:
                return None
        except Exception as e:
            print(e)
            #if(__debug__): print(traceback.format_exc())
            raise Exception(e)

    @staticmethod
    def get_angle_pca(mask):
        if (mask is not None):
            y, x = np.where(mask)
            xy = np.column_stack([x, y])
            centroid = xy.mean(axis=0)
            centroid = [int(centroid[0]), int(centroid[1])]
            xy_diff = xy - centroid
            x_diff, y_diff = xy_diff.T
            cov_matrix = np.cov(x_diff, y_diff)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            sorting_indices = np.argsort(eigenvalues)[::-1]
            _ = eigenvalues[sorting_indices]
            eigenvectors = eigenvectors[:, sorting_indices]

            angle = atan2(-eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
            # calculating degrees for symmetry with phase_cross_correlation output
            return round(np.rad2deg(angle),3),centroid
        else:
            return None
