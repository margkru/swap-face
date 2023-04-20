import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from utils.utils import (estimate_norm, get_lm, inverse_estimate_norm,
                         norm_crop, transform_landmark_points)


def run_inference(source, target, RetinaFace,
                  ArcFace, FaceDancer, result_img_path, source_z=None):
    try:

        if not isinstance(target, str):
            target = target
        else:
            target = cv2.imread(target)

        target = np.array(target)

        if source_z is None:
            source = cv2.imread(source)
            source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            source = np.array(source)

            source_h, source_w, _ = source.shape
            source_a = RetinaFace(np.expand_dims(source, axis=0)).numpy()[0]
            source_lm = get_lm(source_a, source_w, source_h)
            source_aligned = norm_crop(source, source_lm, image_size=112, shrink_factor=1.0)

            source_z = ArcFace.predict(np.expand_dims(source_aligned / 255.0, axis=0))

        blend_mask_base = np.zeros(shape=(256, 256, 1))
        blend_mask_base[77:240, 32:224] = 1
        blend_mask_base = gaussian_filter(blend_mask_base, sigma=7)

        im = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        im_h, im_w, _ = im.shape
        im_shape = (im_w, im_h)

        detection_scale = (im_w // 640) if (im_w > 640) else 1
        faces = RetinaFace(np.expand_dims(cv2.resize(im,
                                                     (im_w // detection_scale,
                                                      im_h // detection_scale)), axis=0)).numpy()
        total_img = im / 255.0

        for annotation in faces:
            lm_align = get_lm(annotation, im_w, im_h)

            # align the detected face
            M, pose_index = estimate_norm(lm_align, 256, "arcface", shrink_factor=1.0)
            im_aligned = cv2.warpAffine(im, M, (256, 256), borderValue=0.0)

            # face swap
            face_swap = FaceDancer.predict([np.expand_dims((im_aligned - 127.5) / 127.5, axis=0), source_z])
            face_swap = (face_swap[0] + 1) / 2

            # get inverse transformation landmarks
            transformed_lmk = transform_landmark_points(M, lm_align)

            # warp image back
            iM, _ = inverse_estimate_norm(lm_align, transformed_lmk, 256, "arcface", shrink_factor=1.0)
            iim_aligned = cv2.warpAffine(face_swap, iM, im_shape, borderValue=0.0)

            # blend swapped face with target image
            blend_mask = cv2.warpAffine(blend_mask_base, iM, im_shape, borderValue=0.0)
            blend_mask = np.expand_dims(blend_mask, axis=-1)

            total_img = (iim_aligned * blend_mask + total_img * (1 - blend_mask))

        total_img = np.clip(total_img * 255, 0, 255).astype('uint8')

        cv2.imwrite(result_img_path, cv2.cvtColor(total_img, cv2.COLOR_BGR2RGB))

        return total_img, source_z

    except Exception as e:
        print('\n', e)
        sys.exit(0)
