from skimage.measure import compare_ssim
import cv2
import cropper

class BboxTrigger:
    def __init__(self, cameraID, refImg, openThresh, closeThresh, chkCoord, sampleCoord, detector):
        self.cameraID = cameraID
        self.openThresh = openThresh
        self.closeThresh = closeThresh
        self.chkCoord = chkCoord
        self.sampleCoord = sampleCoord
        self.detector = detector # ideally this is not here in the future either

        self.refimg = cv2.cvtColor(cropper.crop_image(refImg, chkCoord), cv2.COLOR_BGR2GRAY)

        self.check = 0

    def initialize(self, frames):
        # does nothing right now, but meant to replace refimg assignment in constructor
        pass

    def update(self, frames):
        img = frames[self.cameraID]
        chkimg = cv2.cvtColor(cropper.crop_image(img, self.chkCoord), cv2.COLOR_BGR2GRAY)
        score, diff = compare_ssim(self.refimg, chkimg, full=True)

        if self.check == 1:
            if score > self.closeThresh:
                sampimg = cropper.crop_image(img, self.sampleCoord)
                bboxes, scores = self.detector.get_bboxes(sampimg)
                self.check = 0
                return True, bboxes, sampimg
        else:
            if score < self.openThresh:
                self.check = 1

        return False, None, None

