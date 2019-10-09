import opt
import distancemetrics

threshold = .7

class BasicGallery:
    def __init__(self):
        self.people = list()
        self.ids = list()
        self.distance = distancemetrics.options[opt.args.distance]

    def getID(self, person):
        if len(self.people) == 0:
            self.ids.append(1)
            self.people.append(person)
            return 1

        # calculates distance between query (person) and stored people
        dists = [self.distance(person, pers) for pers in self.people]
        minval = min(dists)
        if minval > threshold:
            return self.ids[dists.index(minval)]
        else:
            id = max(self.ids) + 1
            self.ids.append(id)
            self.people.append(person)
            return id


options = {
    opt.defaultkey: BasicGallery,
    "basic": BasicGallery
}