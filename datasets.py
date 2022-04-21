import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list

class DataProvider:
    VALID_SEED = 0  # random seed for the validation set

    @staticmethod
    def name():
        """ Return name of the dataset """
        raise NotImplementedError

    @property
    def data_shape(self):
        """ Return shape as python list of one data entry """
        raise NotImplementedError

    @property
    def n_classes(self):
        """ Return `int` of num classes """
        raise NotImplementedError

    @property
    def save_path(self):
        """ local path to save the data """
        raise NotImplementedError

    @property
    def data_url(self):
        """ link to download the data """
        raise NotImplementedError

    @staticmethod
    def random_sample_valid_set(train_labels, valid_size, n_classes):
        train_size = len(train_labels)
        assert train_size > valid_size

        g = torch.Generator()
        g.manual_seed(DataProvider.VALID_SEED)  # set random seed before sampling validation set
        rand_indexes = torch.randperm(train_size, generator=g).tolist()

        train_indexes, valid_indexes = [], []
        per_class_remain = get_split_list(valid_size, n_classes)

        for idx in rand_indexes:
            label = train_labels[idx]
            if isinstance(label, float):
                label = int(label)
            elif isinstance(label, np.ndarray):
                label = np.argmax(label)
            else:
                assert isinstance(label, int)
            if per_class_remain[label] > 0:
                valid_indexes.append(idx)
                per_class_remain[label] -= 1
            else:
                train_indexes.append(idx)
        return train_indexes, valid_indexes


class ImageDatasetProvider(DataProvider):

    def __init__(self, dataset_type='imagenet', save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=16, resize_scale=0.08, distort_color=None):

        self._save_path = save_path
        self._dataset_type = dataset_type
        if dataset_type == 'imagenet':
            train_transforms = self.build_train_transform(distort_color, resize_scale)
            train_dataset = datasets.ImageFolder(self.train_path, train_transforms)
        elif dataset_type == 'cifar10':
            train_transforms = self.build_train_transform(distort_color, resize_scale, cifar10_mode=True)
            train_dataset = datasets.CIFAR10(root=self.train_path, train=True, transform=train_transforms, download=True)
        elif dataset_type == 'cifar100':
            train_transforms = self.build_train_transform(distort_color, resize_scale, cifar10_mode=True)
            train_dataset = datasets.CIFAR100(root=self.train_path, train=True, transform=train_transforms, download=True)
        else:
            raise ValueError(f"Dataset '{dataset_type}' not supported!")

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
                
            if dataset_type == 'imagenet':
                train_indexes, valid_indexes = self.random_sample_valid_set(
                    [cls for _, cls in train_dataset.samples], valid_size, self.n_classes,
                )
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

                valid_dataset = datasets.ImageFolder(self.train_path, transforms.Compose([
                    transforms.Resize(self.resize_value),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    self.normalize,
                ]))
                
                self.train = torch.utils.data.DataLoader(
                    train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                    num_workers=n_worker, pin_memory=True,
                )
                self.valid = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                    num_workers=n_worker, pin_memory=True,
                )
            elif dataset_type == 'cifar10':
                valid_dataset = datasets.CIFAR10(root=self.train_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor()
                ]), download=True)
                
                self.train = torch.utils.data.DataLoader(
                    train_dataset, batch_size=train_batch_size, shuffle=True,
                    num_workers=n_worker, pin_memory=True,
                )
                self.valid = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=test_batch_size,
                    num_workers=n_worker, pin_memory=True,
                )
            elif dataset_type == 'cifar100':
                valid_dataset = datasets.CIFAR100(root=self.train_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor()
                ]), download=True)

                self.train = torch.utils.data.DataLoader(
                    train_dataset, batch_size=train_batch_size, shuffle=True,
                    num_workers=n_worker, pin_memory=True,
                )
                self.valid = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=test_batch_size,
                    num_workers=n_worker, pin_memory=True,
                )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True
            )
            self.valid = None

        if dataset_type == 'imagenet':
            self.test = torch.utils.data.DataLoader(
                datasets.ImageFolder(self.valid_path, transforms.Compose([
                    transforms.Resize(self.resize_value),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    self.normalize,
                ])), batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
            )
        elif dataset_type == 'cifar10':
            self.test = torch.utils.data.DataLoader(
                datasets.CIFAR10(root=self.train_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor()
                ]), download=True), batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
            )
        elif dataset_type == 'cifar100':
            self.test = torch.utils.data.DataLoader(
                datasets.CIFAR100(root=self.train_path, train=False, transform=transforms.Compose([
                    transforms.ToTensor()
                ]), download=True), batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
            )

        if self.valid is None:
            self.valid = self.test

    @property
    def name(self):
        return self._dataset_type

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        if self._dataset_type == 'imagenet':
            return 1000
        elif self._dataset_type == 'cifar10':
            return 10
        elif self._dataset_type == 'cifar100':
            return 100

    @property
    def save_path(self):
        if self._dataset_type == 'imagenet':
            if self._save_path is None:
                self._save_path = '/dataset/imagenet'
                print(f"Forcing save_path to {self._save_path}")
        else:
            if self._save_path is None:
                self._save_path = os.path.join(os.getcwd(), 'dataset')
                print(f"Forcing save_path to {self._save_path}")
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download ImageNet')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def build_train_transform(self, distort_color, resize_scale, cifar10_mode=False):
        print('Color jitter: %s' % distort_color)
        if distort_color == 'strong':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif distort_color == 'normal':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None
        if color_transform is None:
            transforms_list = [
                transforms.RandomRotation(45),
                transforms.RandomGrayscale(0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
            print("Additional Transforms added")
            if not cifar10_mode:
                transforms_list = [
                    transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                    *transforms_list,
                    self.normalize
                ]
            
            train_transforms = transforms.Compose(transforms_list)
        else:
            transforms_list = [
                transforms.RandomRotation(45),
                transforms.RandomGrayscale(0.2),
                transforms.RandomHorizontalFlip(),
                color_transform,
                transforms.ToTensor()
            ]
            print("Additional Transforms added")
            
            if not cifar10_mode:
                transforms_list = [
                    transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                    *transforms_list,
                    self.normalize
                ]
            
            train_transforms = transforms.Compose(transforms_list)
        return train_transforms

    @property
    def resize_value(self):
        return 256

    @property
    def image_size(self):
        if self._dataset_type == 'imagenet':
            return 224
        else:
            return 32