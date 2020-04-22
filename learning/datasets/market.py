import torch
import torchvision
import os
import torch.utils.data as data
from torchvision.datasets.utils import download_and_extract_archive, download_url
import glob
import numpy as np
import scipy.io
from PIL import Image
from collections import defaultdict
from functools import partial

ATTRIBUTE_LABEL = {
    'age': {
        1: "young",
        2: "teenage",
        3: "adult",
        4: "old"
    },
    'backpack': {
        1: "no",
        2: "yes"
    },
    'bag': {
        1: "no",
        2: "yes"
    },
    'handbag': {
        1: "no",
        2: "yes"
    },
    'clothes': {
        1: "dress",
        2: "pants"
    },
    'up': {
        1: "long sleeve",
        2: "short sleeve"
    },
    'down': {
        1: "long lower body clothing",
        2: "short"
    },
    'hair': {
        1: "short hair",
        2: "long hair"
    },
    'hat': {
        1: "no",
        2: "yes"
    },
    'gender': {
        1: "male",
        2: "female"
    },
    'upblack': {
        1: "no",
        2: "yes"
    },
    'upwhite': {
        1: "no",
        2: "yes"
    },
    'upred': {
        1: "no",
        2: "yes"
    },
    'uppurple': {
        1: "no",
        2: "yes"
    },
    'upyellow': {
        1: "no",
        2: "yes"
    },
    'upgray': {
        1: "no",
        2: "yes"
    },
    'upblue': {
        1: "no",
        2: "yes"
    },
    'upgreen': {
        1: "no",
        2: "yes"
    },
    'downblack': {
        1: "no",
        2: "yes"
    },
    'downwhite': {
        1: "no",
        2: "yes"
    },
    'downpink': {
        1: "no",
        2: "yes"
    },
    'downpurple': {
        1: "no",
        2: "yes"
    },
    'downyellow': {
        1: "no",
        2: "yes"
    },
    'downgray': {
        1: "no",
        2: "yes"
    },
    'downblue': {
        1: "no",
        2: "yes"
    },
    'downgreen': {
        1: "no",
        2: "yes"
    },
    'downbrown': {
        1: "no",
        2: "yes"
    }
}

ATTRIBUTES = sorted(ATTRIBUTE_LABEL.keys())


def _canonicalize_attributes_mat(attributes, set_name):
    # This function is to fix the bizzare and outlandish shape of
    # the attributes tensor saved in the .mat file associated with
    # Market1501 dataset.
    # The original tensor has the shape:
    # [d, d, set_id, d, d, attribute_id, d, person_idx]
    # where d stands for a "don't care" dimension that is always of size 1
    # The resulting tensor from this function has the shape
    # [set_id, attribute_id, person_idx] serving a more sane indexing code.
    canonical_mat = np.zeros(shape=(1, Market1501Dataset.num_attributes + 1,
                                    Market1501Dataset.num_ids[set_name]))
    for attribute_id, attribute in enumerate(ATTRIBUTES + ['image_index']):
        canonical_mat[0, attribute_id] = attributes['market_attribute'][0][0][
            set_name][0][0][attribute][0]
    return canonical_mat


class Market1501Dataset(data.Dataset):

    dataset_dir_name = "Market-1501-v15.09.15"
    dataset_url = "http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip"
    attributes_url = "https://raw.githubusercontent.com/vana77/Market-1501_Attribute/master/market_attribute.mat"
    num_attributes = 27
    num_ids = {'train': 751, 'test': 750}
    train_path = "bounding_box_train"
    test_path = "bounding_box_test"
    # TODO: (nhendy) figure out this gt bbox business
    gt_path = "gt_bbox"
    attributes_file = "market_attribute.mat"

    def __init__(self,
                 root,
                 input_transforms=None,
                 target_transforms=None,
                 train=True,
                 include_attributes_labels=True):
        super(Market1501Dataset, self).__init__()
        self.root = root
        self.train = train
        self._include_attributes = include_attributes_labels
        self._input_transform = input_transforms
        self._target_transform = target_transforms
        self._download()
        self._load_imgs()
        self._make_inputs()
        self._make_attributes()
        self._make_targets()
        if include_attributes_labels:
            self._filter_inputs_targets()

    def _set_id_to_name(self, id):
        return 'train' if id == 1 else 'test'

    def _requested_set_name(self):
        return 'train' if self.train else 'test'

    def _make_attributes(self):
        attributes_mat = scipy.io.loadmat(
            os.path.join(self.root, self.dataset_dir_name,
                         self.attributes_file))
        attributes_mat = _canonicalize_attributes_mat(
            attributes_mat, self._requested_set_name())
        self._person_id_to_attributes = {}
        for attribute_id in range(attributes_mat.shape[1] - 1):
            for person_idx in range(attributes_mat.shape[2]):
                set_name = self._requested_set_name()
                if person_idx >= self.num_ids[set_name]:
                    continue
                person_id = int(attributes_mat[0, -1, person_idx])
                if person_id not in self._person_id_to_attributes.keys():
                    self._person_id_to_attributes[person_id] = np.zeros((27, ))
                self._person_id_to_attributes[person_id][
                    attribute_id] = attributes_mat[0, attribute_id, person_idx]

    def _download(self):
        download_and_extract_archive(self.dataset_url, self.root)
        download_url(self.attributes_url,
                     os.path.join(self.root, self.dataset_dir_name))

    def _load_imgs(self):
        imgs_path = os.path.join(
            self.root, self.dataset_dir_name,
            self.train_path if self.train else self.test_path)
        self._imgs = glob.glob(os.path.join(imgs_path, "*.jpg"))

    def _make_targets(self):
        self._target_person_ids = []
        for img in self._imgs:
            basename = os.path.basename(img)
            self._target_person_ids.append(int(basename.split('_')[0]))

    def _make_inputs(self):
        self._inputs = []
        for img in self._imgs:
            self._inputs.append(np.array(Image.open(img)))

    def represent_label(self, label):
        if not isinstance(label, np.ndarray):
            raise ValueError(
                "It must be a numpy array of shape (28,), first element is person id"
            )
        if not label.shape == (28, ):
            raise ValueError(
                "It must be a numpy array of shape (28,), first element is person id"
            )
        repr_str = []
        for idx, attribute in enumerate(label[1:]):
            attribute_name = ATTRIBUTES[idx]
            str_label = ATTRIBUTE_LABEL[attribute_name][attribute]
            repr_str.append("{}: {}".format(attribute_name, str_label))
        return "Person id: {} ".format(label[0]) + ", ".join(repr_str)

    def _filter_inputs_targets(self):
        filtered_imgs = []
        filtered_targets = []
        set_name = self._requested_set_name()
        for i, person_id in enumerate(self._target_person_ids):
            if person_id in self._person_id_to_attributes.keys():
                filtered_imgs.append(self._inputs[i])
                filtered_targets.append(self._target_person_ids[i])
        self._inputs = filtered_imgs
        self._target_person_ids = filtered_targets

    def __getitem__(self, index):
        img, target = self._inputs[index], self._target_person_ids[index]
        if self._include_attributes:
            set_name = self._requested_set_name()
            attributes = self._person_id_to_attributes[target]
            target = np.concatenate(
                [np.asarray(target).reshape((1, )),
                 attributes]).astype(np.float32)
        if self._input_transform:
            img = self._input_transform(img)
        if self._target_transform:
            target = self._target_transform(target)
        return img, target

    def __len__(self):
        return len(self._inputs)
