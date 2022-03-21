import SimpleITK as sitk
import numpy as np


def read_landmarks(filepath):
    with open(filepath) as f:
        lines = [tuple(map(float, line.rstrip().split('\t'))) for line in f]
    return lines


def transform_landmarks(point_list, reference_image):
    return [reference_image.TransformContinuousIndexToPhysicalPoint(p) for p in point_list]


def target_registration_errors(tx, point_list, reference_point_list):
    """
    Distances between points transformed by the given transformation and their
    location in another coordinate system. When the points are only used to evaluate
    registration accuracy (not used in the registration) this is the target registration
    error (TRE).
    """
    return [np.linalg.norm(np.array(tx.TransformPoint(p)) - np.array(p_ref))
            for p, p_ref in zip(point_list, reference_point_list)]

def landmark_distance(point_list, reference_point_list):
    return [np.linalg.norm(np.array(p) - np.array(p_ref))
            for p, p_ref in zip(point_list, reference_point_list)]

def load_and_preprocess(filepath):
    filepath = str(filepath)
    image = sitk.ReadImage(filepath, sitk.sitkFloat32)
    image = sitk.GetArrayFromImage(image)
    return image
