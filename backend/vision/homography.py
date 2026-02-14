# models/homography.py

import cv2
import numpy as np
from scipy.spatial import distance

from backend.vision.court_reference import CourtReference


class HomographyEstimator:
    """
    Computes the homography between the canonical tennis court reference
    and the image frame using detected court keypoints.
    """

    def __init__(self):
        # Load canonical court reference
        self.court_ref = CourtReference()

        # Reference keypoints in court coordinate system
        self.refer_kps = (
            np.array(self.court_ref.key_points, dtype=np.float32)
            .reshape((-1, 1, 2))
        )

        # Precompute configuration → keypoint indices
        self.court_conf_ind = self._build_conf_indices()

    def _build_conf_indices(self):
        """
        Map court configurations to indices in key_points list.
        """
        conf_indices = {}

        for conf_id, conf_points in self.court_ref.court_conf.items():
            inds = []
            for pt in conf_points:
                inds.append(self.court_ref.key_points.index(pt))
            conf_indices[conf_id] = inds

        return conf_indices

    def estimate(self, detected_kps):
        """
        Returns:
            H_ref: frame -> reference court
            H_frame: reference court -> frame
        """
        H_frame = self.get_trans_matrix(detected_kps)  # court → frame
        if H_frame is None:
            return None, None

        ok, H_ref = cv2.invert(H_frame)
        if not ok:
            return None, None

        return H_ref, H_frame
    
    def get_trans_matrix(self, points):
        """
        Determine the best homography matrix from detected court points.

        Parameters
        ----------
        points : list[(x, y) | None]
            Detected court keypoints in image space (length = 14)

        Returns
        -------
        H : np.ndarray | None
            Homography matrix mapping COURT_REFERENCE → IMAGE
        """

        best_H = None
        best_error = np.inf

        for conf_id, conf_points in self.court_ref.court_conf.items():
            inds = self.court_conf_ind[conf_id]

            # Extract the 4 required detected points
            image_pts = [points[i] for i in inds]

            if any(p is None for p in image_pts):
                continue

            image_pts = np.array(image_pts, dtype=np.float32)
            ref_pts = np.array(conf_points, dtype=np.float32)

            # Estimate homography
            H, _ = cv2.findHomography(ref_pts, image_pts, method=0)
            if H is None:
                continue

            # Project all reference keypoints
            try:
                projected = cv2.perspectiveTransform(self.refer_kps, H)
                projected = projected.squeeze(1)
            except cv2.error:
                continue

            # Compute reprojection error on remaining points
            errors = []
            for i, pt in enumerate(points):
                if pt is None or i in inds:
                    continue
                errors.append(distance.euclidean(pt, projected[i]))

            if not errors:
                mean_error = 0.0
            else:
                mean_error = float(np.mean(errors))

            if mean_error < best_error:
                best_error = mean_error
                best_H = H

        return best_H

    @staticmethod
    def invert_homography(H):
        """
        Safely invert a homography matrix.
        """
        if H is None:
            return None
        ok, H_inv = cv2.invert(H)
        return H_inv if ok else None
