from skimage.measure import compare_ssim
import cv2
from utils import crop_image
import collections

TriggerContext = collections.namedtuple(
    "TriggerContext", ["triggered_flag", "bboxes", "sample_img"])


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

        self._door_opened = False

    def update(self, frame):
        img = frame
        chkimg = cv2.cvtColor(crop_image(img, self._check_coords),
                              cv2.COLOR_BGR2GRAY)
        score, diff = compare_ssim(self._ref_img, chkimg, full=True)

        if self._door_opened:
            if score > self._close_thresh:
                sampimg = crop_image(img, self._sample_coords)
                bboxes, scores = self._detector.get_bboxes(sampimg)
                self._door_opened = False
                return TriggerContext(triggered_flag=True,
                                      bboxes=bboxes,
                                      sample_img=sampimg)
        else:
            if score < self._open_thresh:
                self._door_opened = True

        return TriggerContext(triggered_flag=False,
                              bboxes=None,
                              sample_img=None)
