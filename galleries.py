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
