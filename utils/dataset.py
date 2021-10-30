import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    '''
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    '''

    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_dataloader(args):
    '''
    Returns a dictionary of training and validation dataloaders
    '''
    
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    TRAIN_DIR_PATH = os.path.join(args.data_dir, "train")
    VAL_DIR_PATH = os.path.join(args.data_dir, "val")

    train_dataset = ImageFolderWithPaths(TRAIN_DIR_PATH,transform=transformations)
    val_dataset = ImageFolderWithPaths(VAL_DIR_PATH,transform=transformations)

    print(f'Length of train dataset: {len(train_dataset)} \nLength of validation dataset: {len(val_dataset)}')

    data_loader = {
        'training' : DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4),
        'validation' : DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)
    }

    return train_dataset.class_to_idx, data_loader
