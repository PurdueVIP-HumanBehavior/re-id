from constants import defaultkey
from utils import unitdotprod
from utils import crop_image
import numpy as np


class TriggerGallery:
    """
    this class is the gallery for images obtained from BboxTrigger

    Attributes:
    _people (list): list of people in the gallery
    _feats (list): feature vector corresponding to each person in the gallery
    _triggers (BboxTrigger): bounding box trigger object
    _attribute_extractor (MgnWrapper): Attribute extractor
    """
    def __init__(self, attribute_extractor, triggers):
        """
        constructor for TriggerGallery

        Parameters:
        triggers (BboxTrigger): bounding box trigger object
        attribute_extractor (MgnWrapper): Attribute extractor
        """
        self._people = list()
        self._feats = list()
        self._triggers = triggers
        self._attribute_extractor = attribute_extractor

    def add_trigger(self, trig):
        """adds a trigger"""
        self._triggers.append(trig)

    def update(self, frames):
        """adds people and features to the gallery"""
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
        """Returns list of people"""
        return self._people

class TriggerLineGallery:
    """
    this class is the gallery for images obtained from VectorTrigger

    Attributes:
    _people (list): list of people in the gallery
    _feats (list): feature vector corresponding to each person in the gallery
    _triggers (BboxTrigger): bounding box trigger object
    _attribute_extractor (MgnWrapper): Attribute extractor
    """
    def __init__(self, attribute_extractor, triggers):
        """
        constructor for TriggerGallery

        Parameters:
        triggers (BboxTrigger): bounding box trigger object
        attribute_extractor (MgnWrapper): Attribute extractor
        """
        self.triggers = triggers
        self._people = list()
        self._feats = list()
        self._attribute_extractor = attribute_extractor

    def update(self, video_name, frame, bboxes):
        """adds people and features to the gallery"""
        for trig in self.triggers:
            if trig.video_oi != video_name: continue
            indsToSave = trig.update(bboxes)
            for index in indsToSave:
                cropimg = crop_image(frame, np.reshape(bboxes[index, :-1], (2,2)))
                vect = self._attribute_extractor(cropimg)
                self._people.append(cropimg)
                self._feats.append(vect)

    def people(self):
        """Returns list of people"""
        return self._people