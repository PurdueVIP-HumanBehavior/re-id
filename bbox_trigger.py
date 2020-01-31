from skimage.measure import compare_ssim
import cv2
import cropper


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
        self._ref_img = cv2.cvtColor(cropper.crop_image(ref_img, check_coords),
                                     cv2.COLOR_BGR2GRAY)

        self._check = 0

    def update(self, frames):
        img = frames[self.camera_id]
        chkimg = cv2.cvtColor(cropper.crop_image(img, self.check_coords),
                              cv2.COLOR_BGR2GRAY)
        score, diff = compare_ssim(self.ref_img, chkimg, full=True)

        if self._check == 1:
            if score > self._close_thresh:
                sampimg = cropper.crop_image(img, self._sample_coords)
                bboxes, scores = self._detector.getBboxes(sampimg)
                self._check = 0
                return True, bboxes, sampimg
        else:
            if score < self._open_thresh:
                self._check = 1

        return False, None, None
