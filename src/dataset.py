"""
dataset
"""
import os
import glob
import numpy as np
from PIL import Image
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose


class MVTecDataset():
    """
    MVTecDataset
    """
    def __init__(self, root, transform, gt_transform, phase, is_json=False):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')

        self.is_json = is_json
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def load_dataset(self):
        """load_dataset
        """
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)[0]

        if gt == 0:
            gt = np.zeros((1, np.array(img).shape[-2], np.array(img).shape[-2])).tolist()
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)[0]

        if self.is_json:
            return os.path.basename(img_path[:-4]), img_type
        return img, gt, label, idx


def createDataset(dataset_path, category):
    """createDataset
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = Compose([
        py_vision.Resize((256, 256), interpolation=Inter.ANTIALIAS),
        py_vision.CenterCrop(256),
        py_vision.ToTensor(),
        py_vision.Normalize(mean=mean, std=std)
    ])
    gt_transforms = Compose([
        py_vision.Resize((256, 256)),
        py_vision.CenterCrop(256),
        py_vision.ToTensor()
    ])

    train_data = MVTecDataset(root=os.path.join(dataset_path, category),
                              transform=data_transforms, gt_transform=gt_transforms, phase='train')
    test_data = MVTecDataset(root=os.path.join(dataset_path, category),
                             transform=data_transforms, gt_transform=gt_transforms, phase='test')

    train_dataset = ds.GeneratorDataset(train_data, column_names=['img', 'gt', 'label', 'idx'],
                                        shuffle=True)
    test_dataset = ds.GeneratorDataset(test_data, column_names=['img', 'gt', 'label', 'idx'],
                                       shuffle=False)

    train_dataset = train_dataset.batch(32, drop_remainder=False)
    test_dataset = test_dataset.batch(1, drop_remainder=True)

    return train_dataset, test_dataset
