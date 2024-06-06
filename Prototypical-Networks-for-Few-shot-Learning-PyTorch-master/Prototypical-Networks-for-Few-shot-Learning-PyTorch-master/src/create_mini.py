import logging
import os

import numpy as np

def _load_mini_imagenet(data_dir, split):
    """Load mini-imagenet from numpy's npz file format."""
    _split_tag = {'sources': 'train', 'target_val': 'val', 'target_tst': 'test'}[split]
    dataset_path = os.path.join(data_dir, 'few-shot-{}.npz'.format(_split_tag))
    logging.info("Loading mini-imagenet...")
    data = np.load(dataset_path)
    fields = data['features'], data['targets']
    logging.info("Done loading.")
    return fields
def get_image_size(data_dir):
    if 'mini-imagenet' or 'tiered' in data_dir:
        image_size = 84
    elif 'cifar' in data_dir:
        image_size = 32
    else:
        raise Exception('Unknown dataset: %s' % data_dir)
    return image_size