from constants import defaultkey
from utils import unitdotprod
from utils import crop_image


class TriggerGallery:
    def __init__(self, attribute_extractor, triggers):
        self._people = list()
        self._feats = list()
        self._triggers = triggers
        self._attribute_extractor = attribute_extractor

    def add_trigger(self, trig):
        self._triggers.append(trig)

    def update(self, frames):
        for trig in self._triggers:
            add, boxes, img = trig.update(frames)
            if add:
                for box in boxes:
                    cropimg = crop_image(img, box)
                    vect = self._attribute_extractor(cropimg)
                    self._people.append(cropimg)
                    self._feats.append(vect)

    @property
    def people(self):
        return self._people

class TriggerLineGallery:
    def __init__(self, attribute_extractor, triggers):
        self.triggers = triggers
        self._people = list()
        self._feats = list()
        self._attribute_extractor = attribute_extractor

    def update(self, video_name, frame, bboxes):
        for trig in self.triggers:
            if trig.video_oi != video_name: continue
            indsToSave = trig.update(bboxes)
            for index in indsToSave:
                cropimg = crop_image(frame, bboxes[index, 1:])
                vect = self._attribute_extractor(cropimg)
                self._people.append(cropimg)
                self._feats.append(vect)

    def people(self):
        return self._people