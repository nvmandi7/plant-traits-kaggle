
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import transforms

class TransformHolder:

    @staticmethod
    def get_base_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @staticmethod
    def get_train_transform():
        return A.Compose([
            A.Resize(256, 256),  # Resize images to a common size
            A.HorizontalFlip(p=0.5),  # Apply horizontal flip with a probability of 0.5
            A.VerticalFlip(p=0.5),  # Apply vertical flip with a probability of 0.5
            A.RandomRotate90(p=0.5),  # Randomly rotate the image by 90 degrees
            A.RandomBrightnessContrast(p=0.2),  # Adjust brightness and contrast
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),  # Randomly change hue and saturation
            A.GaussNoise(p=0.2),  # Add random Gaussian noise
            A.Normalize(),  # Normalize pixel values to be in the range [0, 1]
            ToTensorV2(),  # Convert the image to a PyTorch tensor
        ])

    @staticmethod
    def get_val_transform():
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2(),
        ])

