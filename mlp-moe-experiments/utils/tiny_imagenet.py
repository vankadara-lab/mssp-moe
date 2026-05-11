import os
import urllib.request
import zipfile
from pathlib import Path
from torchvision import datasets
from PIL import Image

class TinyImageNet(datasets.ImageFolder):
    """TinyImageNet dataset with automatic download and setup"""
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = Path(root)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. Use download=True to download it.')

        # Set the correct directory
        if train:
            # Format training set if needed (move images from images/ subdirectory)
            self._format_train()
            data_dir = self.root / 'tiny-imagenet-200' / 'train_formatted'
        else:
            data_dir = self.root / 'tiny-imagenet-200' / 'val_formatted'
            if not data_dir.exists():
                self._format_val()

        super().__init__(root=str(data_dir), transform=transform)
    
    def _check_exists(self):
        return (self.root / 'tiny-imagenet-200').exists()
    
    def download(self):
        if self._check_exists():
            print('TinyImageNet already downloaded')
            return
        
        print('Downloading TinyImageNet...')
        self.root.mkdir(parents=True, exist_ok=True)
        zip_path = self.root / 'tiny-imagenet-200.zip'
        
        urllib.request.urlretrieve(self.url, zip_path)
        
        print('Extracting...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        
        zip_path.unlink()
        print('Download complete')
    
    def _format_train(self):
        """Format training set to ImageFolder structure (move images out of images/ subdirectory)"""
        train_dir = self.root / 'tiny-imagenet-200' / 'train'
        train_formatted = self.root / 'tiny-imagenet-200' / 'train_formatted'

        if train_formatted.exists():
            return

        print('Formatting training set...')
        train_formatted.mkdir(exist_ok=True)

        # Iterate through each class directory
        for class_dir in train_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            images_dir = class_dir / 'images'

            if not images_dir.exists():
                continue

            # Create class directory in formatted location
            formatted_class_dir = train_formatted / class_name
            formatted_class_dir.mkdir(exist_ok=True)

            # Move all images from images/ subdirectory to class directory
            for img_file in images_dir.iterdir():
                if img_file.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                    dst = formatted_class_dir / img_file.name
                    if not dst.exists():
                        img_file.rename(dst)

        print('Training set formatted')

    def _format_val(self):
        """Format validation set to ImageFolder structure"""
        val_dir = self.root / 'tiny-imagenet-200' / 'val'
        val_formatted = self.root / 'tiny-imagenet-200' / 'val_formatted'

        if val_formatted.exists():
            return

        print('Formatting validation set...')
        val_formatted.mkdir(exist_ok=True)

        # Read val annotations
        val_annotations = val_dir / 'val_annotations.txt'
        with open(val_annotations, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name, class_name = parts[0], parts[1]

                # Create class directory
                class_dir = val_formatted / class_name
                class_dir.mkdir(exist_ok=True)

                # Move image
                src = val_dir / 'images' / img_name
                dst = class_dir / img_name
                if src.exists() and not dst.exists():
                    src.rename(dst)

        print('Validation set formatted')
