from torch.utils.data import Dataset, DataLoader	
from torchvision import datasets, transforms	
import torch	

class GaussianDataset(Dataset):	
    """Face Landmarks dataset."""	

    def __init__(self, length, size, transform):	
        self.length = length	
        self.transform = transform	
        self.size = size	

    def __len__(self):	
        return self.length	

    def __getitem__(self, idx):	
        # sample = torch.randint(low=0, high=256, size=self.size)	
        # sample = self.transform(sample.float())	
        sample = torch.randn(size = self.size)	
        return sample	

def getGaussianData(name = 'cifar10', num_data = None, batch_size = 512):	
    if name == 'cifar10':	
        size = (3, 32, 32)	
        num_data = 50000 * 10	
    elif name == 'imagenet':	
        size = (3, 224, 224)	
        num_data = 1280000	
    # if num_data is None:	
    #     num_data = 100 * batch_size	
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))	
    gaussian_dataset = GaussianDataset(length=num_data, size=size, transform=normalize)	
    data_loader = DataLoader(gaussian_dataset, batch_size=batch_size, shuffle=False, num_workers=4)	
    return data_loader	

def getData(name = 'imagenet', num_data = None, batch_size = 1024, path=None):	
    if name == 'imagenet':	
        if path is None:	
            path = '~/rscratch0/imagenet12/'	
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],	
                                        std=[0.229, 0.224, 0.225])	
        train_dataset = datasets.ImageFolder(	
        path + 'train',	
        transforms.Compose([	
            transforms.RandomResizedCrop(224),	
            transforms.RandomHorizontalFlip(),	
            transforms.ToTensor(),	
            normalize,	
        ]))	

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)	

        test_dataset = datasets.ImageFolder(	
        path + 'val',	
        transforms.Compose([	
            transforms.Resize(256),	
            transforms.CenterCrop(224),	
            transforms.ToTensor(),	
            normalize,	
        ]))	
        # if num_data is None:	
        #     num_data = len(test_dataset)	
        # test_dataset, _ = torch.utils.data.random_split(test_dataset, [num_data, len(test_dataset) - num_data])	
        test_loader = DataLoader(test_dataset, batch_size = batch_size * 4, shuffle = False, num_workers = 4)	
        return train_loader, test_loader	

    elif name == 'cifar10':	
        data_dir = '~/rscratch0/cifar10/'	
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))	
        transform = transforms.Compose([	
            transforms.RandomCrop(32, padding=4),	
            transforms.RandomHorizontalFlip(),	
            transforms.ColorJitter(brightness=4, contrast=4, saturation=4),	
            transforms.ToTensor(),	
            normalize	
        ])	

        transform_test = transforms.Compose([	
            transforms.ToTensor(),	
            normalize	
        ])	

        train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform)	
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform_test)	

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 32)	
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)	
        return train_loader, test_loader	