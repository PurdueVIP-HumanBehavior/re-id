from skimage.measure import compare_ssim
import cv2
from utils import crop_image
import numpy as np
import collections



class BboxTrigger:
    """
    This class is bounding box trigger for a door to build the gallery

    Attributes:
    _camera_id (str): name of the camera location
    _ref_img (cv2 image): reference image (ex. Closed door)
    _open_thresh (int): threshold value for opening a door
    _close_thresh (int): threshold value for closing a door
    _check_coords (list): 2D list of coordinates to check
    _sample_coords (list): 2D list of trigger coordinates
    _detector (detector.py): object detector (default FasterRCNN)
    _check (int): flag for determining when triggering is in process
    """
    def __init__(self, camera_id, ref_img, open_thresh, close_thresh,
                 check_coords, sample_coords, detector):
        """
        the constructor for BboxTrigger class

        Parameters:
        _camera_id (str): name of the camera location
        _ref_img (cv2 image): reference image (ex. Closed door)
        _open_thresh (int): threshold value for opening a door
        _close_thresh (int): threshold value for closing a door
        _check_coords (list): 2D list of coordinates to check
        _sample_coords (list): 2D list of trigger coordinates
        _detector (detector.py): object detector (default FasterRCNN)

        """
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
        """
        Given a image, find when to trigger and return bounding boxes of people in the trigger region

        Parameters:
        frames (dict): dictionary of frames from all the cameras

        Returns:
        bboxes (ndarry): bounding boxes of detected objects
        sampimg (ndarray): cropped image of the trigger region
        """
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
    """
    This class is for a Line trigger


    """
    def __init__(self, video, vector, inpt, length_thresh, frame_offset):
        tmpvec = np.random.randn(2)
        invec = np.array([vector[2] - vector[0], vector[3] - vector[1]])
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



