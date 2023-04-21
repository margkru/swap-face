import cv2
import numpy as np
from skimage import transform as trans

# facial alignment, taken from https://github.com/deepinsight/insightface
src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
# <--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

# ---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

# -->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

# -->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

# Left eye, right eye, nose, left mouth, right mouth
arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


def estimate_norm(lmk, image_size=112, mode='arcface', shrink_factor=1.0):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    src_factor = image_size / 112
    if mode == 'arcface':
        src = arcface_src * shrink_factor + (1 - shrink_factor) * 56
        src = src * src_factor
    else:
        src = src_map[image_size] * src_factor
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def inverse_estimate_norm(lmk, t_lmk, image_size=112, mode='arcface', shrink_factor=1.0):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    src_factor = image_size / 112
    if mode == 'arcface':
        src = arcface_src * shrink_factor + (1 - shrink_factor) * 56
        src = src * src_factor
    else:
        src = src_map[image_size] * src_factor
    for i in np.arange(src.shape[0]):
        tform.estimate(t_lmk, lmk)
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode='arcface', shrink_factor=1.0):
    """
    Align and crop the image based of the facial landmarks in the image. The alignment is done with
    a similarity transformation based of source coordinates.
    :param img: Image to transform.
    :param landmark: Five landmark coordinates in the image.
    :param image_size: Desired output size after transformation.
    :param mode: 'arcface' aligns the face for the use of Arcface facial recognition model. Useful for
    both facial recognition tasks and face swapping tasks.
    :param shrink_factor: Shrink factor that shrinks the source landmark coordinates. This will include more border
    information around the face. Useful when you want to include more background information when performing face swaps.
    The lower the shrink factor the more of the face is included. Default value 1.0 will align the image to be ready
    for the Arcface recognition model, but usually omits part of the chin. Value of 0.0 would transform all source points
    to the middle of the image, probably rendering the alignment procedure useless.

    If you process the image with a shrink factor of 0.85 and then want to extract the identity embedding with arcface,
    you simply do a central crop of factor 0.85 to yield same cropped result as using shrink factor 1.0. This will
    reduce the resolution, the recommendation is to processed images to output resolutions higher than 112 is using
    Arcface. This will make sure no information is lost by resampling the image after central crop.
    :return: Returns the transformed image.
    """
    M, pose_index = estimate_norm(landmark, image_size, mode, shrink_factor=shrink_factor)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def transform_landmark_points(M, points):
    lmk_tran = np.insert(points, 2, values=np.ones(5), axis=1)
    transformed_lmk = np.dot(M, lmk_tran.T)
    transformed_lmk = transformed_lmk.T

    return transformed_lmk


def get_lm(annotation, im_w, im_h):
    lm_align = np.array([[annotation[4] * im_w, annotation[5] * im_h],
                         [annotation[6] * im_w, annotation[7] * im_h],
                         [annotation[8] * im_w, annotation[9] * im_h],
                         [annotation[10] * im_w, annotation[11] * im_h],
                         [annotation[12] * im_w, annotation[13] * im_h]],
                        dtype=np.float32)
    return lm_align
