import torch
from torch.utils import data
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
class CustomDataset(data.Dataset):
    def __init__(self, dataset, indices, source_class = None, target_class = None):
        self.dataset = dataset
        self.indices = indices
        self.source_class = source_class
        self.target_class = target_class  
        self.contains_source_class = False
            
    def __getitem__(self, index):
        x, y = self.dataset[int(self.indices[index])][0], self.dataset[int(self.indices[index])][1]
        if y == self.source_class:
            y = self.target_class 
        return x, y 

    def __len__(self):
        return len(self.indices)

class PoisonedDataset(data.Dataset):
    def __init__(self, dataset, source_class = None, target_class = None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class  
            
    def __getitem__(self, index):
        x, y = self.dataset[index][0], self.dataset[index][1]
        if y == self.source_class:
            y = self.target_class 
        return x, y 

    def __len__(self):
        return len(self.dataset)

    
class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        Argument:
        reviews: a numpy array
        targets: a vector array
        
        Return xtrain and ylabel in torch tensor datatype
        """
        self.reviews = reviews
        self.target = targets
    
    def __len__(self):
        # return length of dataset
        return len(self.reviews)
    
    def __getitem__(self, index):
        # given an index (item), return review and target of that index in torch tensor
        x = torch.tensor(self.reviews[index,:], dtype = torch.long)
        y = torch.tensor(self.target[index], dtype = torch.float)
        
        return  x, y

# A method for combining datasets  
def combine_datasets(list_of_datasets):
    return data.ConcatDataset(list_of_datasets)



class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['Path'])  # Make sure 'Path' column is relative
        label = int(row['ClassId'])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

def get_gtsrb():
    train_csv = 'data/gtsrb/Train.csv'
    test_csv = 'data/gtsrb/Test.csv'
    root_dir = 'data/gtsrb'  # This is the base path for all images

    trainset = GTSRBDataset(train_csv, root_dir)
    testset = GTSRBDataset(test_csv, root_dir)
    return trainset, testset
class MineSignsDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = img_path.replace(".jpg", ".txt")

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        with open(label_path) as f:
            line = f.readline().strip()
            cls, xc, yc, w, h = map(float, line.split())

        # Convert YOLO → pixel bbox
        x1 = int((xc - w/2) * W)
        y1 = int((yc - h/2) * H)
        x2 = int((xc + w/2) * W)
        y2 = int((yc + h/2) * H)

        cropped = image.crop((x1, y1, x2, y2))

        if self.transform:
            cropped = self.transform(cropped)

        return cropped, int(cls)
class MineSignDataset2:

        def __init__(self, root_dir, batch_size=32):

            self.root_dir = root_dir
            self.batch_size = batch_size

            self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        def get_train_loader(self):

            train_path = os.path.join(self.root_dir, "train")

            train_dataset = ImageFolder(
            root=train_path,
            transform=self.transform
        )

            return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        def get_val_loader(self):

            val_path = os.path.join(self.root_dir, "val")

            val_dataset = ImageFolder(
            root=val_path,
            transform=self.transform
        )

            return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
def get_minesigns():
    root = "S:/Summer25/MineDataset/Annotation_Done/MNIST_Format"  # has train/ and val/

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    trainset = ImageFolder(root + "/train", transform=transform)
    testset  = ImageFolder(root + "/val",   transform=transform)

    return trainset, testset





class DamagedSignsDataset(Dataset):
    """
    One-label-per-image multiclass dataset stored as:
      - images in folder
      - CSV with columns: filename, <class1>, <class2>, ... (one-hot)
    """
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        if "filename" not in self.df.columns:
            raise ValueError(f"'filename' column not found in {csv_path}")

        self.label_cols = [c for c in self.df.columns if c != "filename"]
        if len(self.label_cols) < 2:
            raise ValueError("Expected >=2 label columns (one-hot).")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["filename"]
        img_path = os.path.join(self.img_dir, fname)

        image = Image.open(img_path).convert("RGB")

        # one-hot -> class index
        y = row[self.label_cols].to_numpy(dtype="float32")
        label = int(y.argmax())

        if self.transform:
            image = self.transform(image)

        return image, label




def get_damaged_signs():
    # CHANGE THIS ROOT to your dataset folder path
    root = "S:/Third_Paper_LF_Attack/Dataset/traffic_dataset"

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # CHANGE THESE CSV NAMES if yours are different
    train_csv = os.path.join(root, "train", "_classes.csv")
    valid_csv = os.path.join(root, "valid", "_classes.csv")
    test_csv  = os.path.join(root, "test",  "_classes.csv")

    trainset = DamagedSignsDataset(train_csv, os.path.join(root, "train"), transform=transform)
    testset  = DamagedSignsDataset(test_csv,  os.path.join(root, "test"),  transform=transform)

    return trainset, testset