from constants import defaultkey
import distancemetrics
import cropper

threshold = .7

class BasicGallery:
    def __init__(self):
        self.people = list()
        self.ids = list()
        self.distance = distancemetrics.options[opt.args.distance]

    def get_id(self, person):
        if len(self.people) == 0:
            self.ids.append(1)
            self.people.append(person)
            return 1

        # calculates distance between query (person) and stored people
        dists = [self.distance(person, pers) for pers in self.people]
        minval = max(dists)
        if minval > threshold:
            return self.ids[dists.index(minval)]
        else:
            id = max(self.ids) + 1
            self.ids.append(id)
            self.people.append(person)
            return id

class TriggerGallery:
    def __init__(self, vectFunc):
        self.people = list()
        self.feats = list()
        self.triggers = list()
        self.vectFunc = vectFunc

    def add_trigger(self, trig):
        self.triggers.append(trig)

    def update(self, frames):
        for trig in self.triggers:
            add, boxes, img = trig.update(frames)
            if add:
                for box in boxes:
                    cropimg = cropper.crop_image(img, box)
                    vect = self.vectFunc(cropimg)
                    self.people.append(cropimg)
                    self.feats.append(vect)


options = {
    defaultkey: BasicGallery,
    "basic": BasicGallery
}