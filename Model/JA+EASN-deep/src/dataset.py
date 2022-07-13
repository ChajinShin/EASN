import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib2 import Path


def get_dataloader(opt, is_train=True):
    if is_train:
        dataset = TrainDataset(opt.train_dataset)
        train_data_loader = DataLoader(dataset,
                                 batch_size=opt.train_dataset.batch_size,
                                 shuffle=True,
                                 num_workers=opt.train_dataset.num_workers,
                                 pin_memory=True
                                 )

        dataset = ValDataset(opt.val_dataset)
        val_data_loader = DataLoader(dataset,
                                 batch_size=opt.val_dataset.batch_size,
                                 shuffle=False,
                                 num_workers=opt.val_dataset.num_workers,
                                 pin_memory=True
                                 )
        return train_data_loader, val_data_loader
        
    else:
        dataset = TestDataset(opt.test_dataset)
        test_data_loader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=False)
        return test_data_loader


class TrainDataset(Dataset):
    def __init__(self, data_opt):
        super(TrainDataset, self).__init__()
        transforms_dict = {
            'RandomRotation': transforms.RandomRotation([-180, 180], transforms.InterpolationMode.BICUBIC),
            'RandomCrop': transforms.RandomCrop(size=data_opt.size, pad_if_needed=True),
            'CenterCrop': transforms.CenterCrop(size=data_opt.size),
            'HorizontalFlip': transforms.RandomHorizontalFlip(),
            'VerticalFlip': transforms.RandomVerticalFlip(),
            'ToTensor': transforms.ToTensor()
        }

        transforms_list = list()
        for transform_key in data_opt.augmentation:
            transforms_list.append(transforms_dict[transform_key])
        self.transforms = transforms.Compose(transforms_list)

        self.img_list = list(Path(data_opt.folder_dir).rglob('*.jpg'))\
                        + list(Path(data_opt.folder_dir).rglob('*.jpeg'))\
                        + list(Path(data_opt.folder_dir).rglob('*.png'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        file_path = str(self.img_list[idx])
        img = Image.open(file_path).convert('RGB')
        img = self.transforms(img)
        return img


class ValDataset(Dataset):
    def __init__(self, data_opt):
        super(ValDataset, self).__init__()
        self.transforms = transforms.Compose([
            transforms.CenterCrop(size=data_opt.size),
            transforms.ToTensor()
        ])

        self.img_list = list(Path(data_opt.folder_dir).rglob('*.jpg'))\
                        + list(Path(data_opt.folder_dir).rglob('*.jpeg'))\
                        + list(Path(data_opt.folder_dir).rglob('*.png'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        file_path = str(self.img_list[idx])
        img = Image.open(file_path).convert('RGB')
        img = self.transforms(img)
        return img


class TestDataset(Dataset):
    def __init__(self, data_opt):
        super(TestDataset, self).__init__()
        self.transforms = transforms.ToTensor()

        self.img_list = sorted(list(Path(data_opt.folder_dir).rglob('*.jpg')))\
                        + sorted(list(Path(data_opt.folder_dir).rglob('*.jpeg')))\
                        + sorted(list(Path(data_opt.folder_dir).rglob('*.png')))

        self.img_name = list()
        for img_dir in self.img_list:
            img_base = os.path.basename(str(img_dir))
            fname, ext = os.path.splitext(img_base)
            self.img_name.append(fname)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        file_path = str(self.img_list[idx])
        file_name = self.img_name[idx]
        img = Image.open(file_path).convert('RGB')
        img = self.transforms(img)
        return file_name, img


