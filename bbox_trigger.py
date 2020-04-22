from skimage.measure import compare_ssim
import cv2
from utils import crop_image
import numpy as np
import collections


# TODO: (nhendy) docstring
class BboxTrigger:
    def __init__(self, camera_id, ref_img, open_thresh, close_thresh,
                 check_coords, sample_coords, detector):
        self._camera_id = camera_id
        self._open_thresh = open_thresh
        self._close_thresh = close_thresh
        self._check_coords = check_coords
        self._sample_coords = sample_coords
        self._detector = detector  # ideally this is not here in the future either
        self._ref_img = cv2.cvtColor(crop_image(ref_img, check_coords),
                                     cv2.COLOR_BGR2GRAY)

        self._check = 0

    def update(self, frames):
        img = frames[self._camera_id]
        chkimg = cv2.cvtColor(crop_image(img, self._check_coords),
                              cv2.COLOR_BGR2GRAY)
        score, diff = compare_ssim(self._ref_img, chkimg, full=True)

        if self._check == 1:
            if score > self._close_thresh:
                sampimg = crop_image(img, self._sample_coords)
                bboxes, scores = self._detector.get_bboxes(sampimg)
                self._check = 0
                return True, bboxes, sampimg
        else:
            if score < self._open_thresh:
                self._check = 1

        return False, None, None

"use is completely different from BboxTrigger"
class VectorTrigger:
    def __init__(self, video, vector, inpt, length_thresh, frame_offset):
        tmpvec = np.random.randn(2)
        maxx = max(vector[2], vector[0])
        minx = min(vector[2], vector[0])
        maxy = max(vector[3], vector[1])
        miny = min(vector[3], vector[1])
        invec = np.array([maxx - minx, maxy - miny])
        # tmpvec -= tmpvec.dot(invec) * invec
        # print(tmpvec.dot(invec))
        # self.ovector = tmpvec / np.linalg.norm(tmpvec)
        self.ovector = np.array([invec[1], -1 * invec[0]])
        self.midpt = np.array([(vector[0] + vector[2])/2, (vector[1] + vector[3])/2])
        # 1 is in
        if np.sign(self.ovector.dot(self.midpt - inpt)) != 1: self.ovector = np.zeros(2) - self.ovector
        self.length_thresh = length_thresh
        self.frame_offset = frame_offset

        self.video_oi = video
        self.flags = collections.defaultdict(float)
        self.prev_val = collections.defaultdict(float)

    def update(self, peoplebboxes):
        """
        :param peoplebboxes: numpy array n x 5 with columns in x1, y1, x2, y2, id order
        :return: indexes to capture
        """
        retval = list()
        feetpoints = np.array([(peoplebboxes[:,0] + peoplebboxes[:,2])/2, np.max(peoplebboxes[:, [1,3]], 1)]).transpose()
        displace_vects = self.midpt - feetpoints
        disp_mags = np.linalg.norm(displace_vects, axis=1)

        inout = np.sign(displace_vects.dot(self.ovector))

        for i, (val, bboxes) in enumerate(zip(inout, peoplebboxes)):
            if disp_mags[i] > self.length_thresh:
                continue
            id = bboxes[4]

            if self.flags[id] >= 1: self.flags[id] += 1
            if self.flags[id] >= self.frame_offset:
                retval.append(i)
                self.flags[id] = 0

            # -1 - 1 is entering
            if self.prev_val[id] - val == -2:
                self.flags[id] = 1

            if val == 0:
                self.prev_val[id] = -1
            else: self.prev_val[id] = val

        return retval



