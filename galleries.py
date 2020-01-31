from constants import defaultkey
import distancemetrics
import cropper

# TODO: (nhendy) what is this threshold for?
threshold = .7


class BasicGallery:
    def __init__(self):
        self._people = list()
        self._ids = list()
        self._distance = distancemetrics.options[opt.args.distance]

    def get_id(self, person):
        if len(self._people) == 0:
            self._ids.append(1)
            self._people.append(person)
            # TODO: (nhendy) magic number?
            return 1

        # calculates distance between query (person) and stored people
        dists = [self._distance(person, pers) for pers in self._people]
        minval = max(dists)
        if minval > threshold:
            return self._ids[dists.index(minval)]
        else:
            id = max(self._ids) + 1
            self._ids.append(id)
            self._people.append(person)
            return id


class TriggerGallery:
    def __init__(self, attribute_extractor):
        self._people = list()
        self._feats = list()
        self._triggers = list()
        self._attribute_extractor = attribute_extractor

    def add_trigger(self, trig):
        self._triggers.append(trig)

    def update(self, frames):
        for trig in self._triggers:
            add, boxes, img = trig.update(frames)
            if add:
                for box in boxes:
                    cropimg = cropper.crop_image(img, box)
                    vect = self._attribute_extractor(cropimg)
                    self._people.append(cropimg)
                    self._feats.append(vect)


options = {defaultkey: BasicGallery, "basic": BasicGallery}
