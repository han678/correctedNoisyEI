import os

import chainer
import numpy as np
import six
from chainer.datasets.image_dataset import _read_image_as_array, _postprocess_image, _check_pillow_availability
from chainercv.transforms import center_crop, scale
from chainercv.utils import read_image

# RGB order
_imagenet_mean = np.array(
    [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]


def transform(img):
    """
    img: RGB, CHW
    """
    img = scale(img, size=256)
    img = center_crop(img, size=(224, 224))
    img -= _imagenet_mean
    return img


def transform_with_label(data):
    img, label = data
    img = scale(img, size=256)
    img = center_crop(img, size=(224, 224))
    img -= _imagenet_mean
    return img, label


class ImageDatasetSubset(chainer.dataset.DatasetMixin):

    def __init__(self, paths, root='.', subset=None):
        with open(paths) as paths_file:
            paths = [path.strip() for path in paths_file]
        np.random.shuffle(paths)
        self._paths = paths
        if subset is not None:
            self._paths = self._paths[:subset]
        self._root = root

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        return read_image(path, color=True)  # RGB


class LabeledImageDatasetSubset(chainer.dataset.DatasetMixin):

    def __init__(self, pairs, root='.', dtype=None, label_dtype=np.int32, subset=None):
        _check_pillow_availability()
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    pairs.append((pair[0], int(pair[1])))
        np.random.shuffle(pairs)
        self._pairs = pairs
        if subset is not None:
            self._pairs = self._pairs[:subset]
        self._root = root
        self._dtype = chainer.get_dtype(dtype)
        self._label_dtype = label_dtype

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        image = _read_image_as_array(full_path, self._dtype)

        label = np.array(int_label, dtype=self._label_dtype)
        return _postprocess_image(image), label
