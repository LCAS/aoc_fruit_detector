"""
Started by: Usman Zahidi (uzahidi@lincoln.ac.uk) {08/11/24}

"""
from skimage.transform import warp_polar
from skimage.registration import phase_cross_correlation
import numpy as np
from enum import Enum,unique
import traceback

RADIUS = 300

@unique
class FruitTypes(Enum):
    """
    Enum of different class names
    """
    """
    Strawberry
    """
    Strawberry         = 0
    """
    Tomato
    """
    Tomato              = 1


class FruitOrientation():

    @staticmethod
    def get_angle_from_vector(vector1, vector2,axis):
        norm_vec1=np.linalg.norm(vector1)
        unit_vector1 = vector1 / norm_vec1
        unit_axis = axis / np.linalg.norm(axis)
        dot_product_vector1 = np.dot(unit_vector1, unit_axis)
        angle_vector1 = np.arccos(dot_product_vector1)
        angle_degrees_vector1=angle_vector1

        #UZ: if first eigenvector's unit vector is a gravity anomaly then switch to next
        if (abs(angle_degrees_vector1)<(np.pi/2)):
            return np.rad2deg(angle_vector1),vector1,vector2
        else:
            dot_product_vector1 = np.dot(-unit_vector1, unit_axis)
            angle_vector1 = np.arccos(dot_product_vector1)
            return np.rad2deg(angle_vector1),-vector1,-vector2

    @staticmethod
    def get_angle_pca(mask=None,fruit_type=FruitTypes.Strawberry):
        try:
            if (mask is not None):
                mask=np.asarray(mask)
                y, x = np.where(mask)
                xy = np.column_stack([x, y])
                centroid = xy.mean(axis=0)
                centroid = np.array([np.around(centroid[0]), np.around(centroid[1])])
                com_x,com_y=centroid
                xy_diff = xy - centroid
                x_diff, y_diff = xy_diff.T
                cov_matrix = np.cov(x_diff, y_diff)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                sorting_indices = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[sorting_indices]
                eigenvectors = eigenvectors[:, sorting_indices]
                if fruit_type==FruitTypes.Strawberry:
                    pca0 = -eigenvectors[:, 0] * eigenvalues[0] ** 0.5
                    pca1 = eigenvectors[:, 1] * eigenvalues[1] ** 0.5       #mask negative x-axis, as we're following right-hand rule
                elif fruit_type==FruitTypes.Tomato:
                    #swapping axes due to shape of tomato
                    pca0 = eigenvectors[:, 1] * eigenvalues[1] ** 0.5  # mask negative x-axis, as we're following right-hand rule
                    pca1 = -eigenvectors[:, 0] * eigenvalues[0] ** 0.5
                y_axis = np.array([0, 1])
                theta,vector1,vector2 = FruitOrientation.get_angle_from_vector(pca0, pca1, y_axis)
                if (com_x> (com_x + vector1[0])):
                    theta = np.around(abs(theta),2)
                else:
                    theta = -np.around(abs(theta),2)
                return theta, centroid, vector1,vector2

            else:
                return None
        except Exception as e:
            print(e)
            if(__debug__): print(traceback.format_exc())
            raise Exception(e)


