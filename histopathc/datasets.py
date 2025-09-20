import h5py
import torch
import torch.utils.data
import torchvision
from PIL import Image


import os
import pandas as pd

from .transforms import Staining, AddDust, AddAirBubble, CorruptTransform



def prepare_histopathc_data(dataset_name, data_dir, corruption, original_transforms, batch_size=128, num_workers=1):

    """
    Prepare the specified dataset.

    Parameters:
    ----------
    dataset_name : str
        The name of the dataset to prepare.
    data_dir : str
        The root directory where the dataset is stored.
    corruption : str
        The type of corruption to apply to the dataset, if applicable. 
        Only used for 'cifar10', 'cifar100', and 'tiny-imagenet'.
    batch_size : int, optional
        The number of samples per batch to load. Default is 128.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 1.

    Returns:
    -------
    tuple
        A tuple containing:
        - loader (torch.utils.data.DataLoader): DataLoader for the prepared dataset.
        - dataset (torchvision.datasets or ImageFolder): The prepared dataset.

    """

    if dataset_name == "nct":
        dataset = torchvision.datasets.ImageFolder(root=data_dir)
        conversion_nck = {
                "ADI": "adipose",
                "DEB": "debris",
                "LYM": "lymphocytes",
                "MUC": "mucus",
                "MUS": "smooth muscle",
                "NORM": "normal colon mucosa",
                "STR": "cancer-associated stroma",
                "TUM": "colorectal adenocarcinoma epithelium",
            }
        
        dataset.classes = [conversion_nck[cl] for cl in dataset.classes] 
        dataset.class_to_idx = {conversion_nck[cl]: idx for cl, idx in dataset.class_to_idx.items()}


    elif dataset_name == 'lc25_lung':
        dataset = torchvision.datasets.ImageFolder(root=data_dir)
        conversion_nck = {
                    "lung_aca": "lung adenocarcinoma",
                    "lung_n": "lung benign tissue",
                    "lung_scc": "lung squamous cell carcinoma",
        }
        dataset.classes = [conversion_nck[cl] for cl in dataset.classes]
        dataset.class_to_idx = {conversion_nck[cl]: idx for cl, idx in dataset.class_to_idx.items()}


    elif dataset_name == 'lc25_colon':
        dataset = torchvision.datasets.ImageFolder(root=data_dir)
        conversion_nck = {
                    "colon_aca": "colon adenocarcinoma",
                    "colon_n": "benign colonic tissue",
        }
        dataset.classes = [conversion_nck[cl] for cl in dataset.classes]
        dataset.class_to_idx = {conversion_nck[cl]: idx for cl, idx in dataset.class_to_idx.items()}


    elif dataset_name == 'lc25_all':
        dataset = torchvision.datasets.ImageFolder(root=data_dir)
        conversion_nck = {
                    "lung_aca": "lung adenocarcinoma",
                    "lung_n": "lung benign tissue",
                    "lung_scc": "lung squamous cell carcinoma",
                    "colon_aca": "colon adenocarcinoma",
                    "colon_n": "benign colonic tissue",
        }
        dataset.classes = [conversion_nck[cl] for cl in dataset.classes]
        dataset.class_to_idx = {conversion_nck[cl]: idx for cl, idx in dataset.class_to_idx.items()}


    elif dataset_name == 'skin':
        dataset = SkinDataset(root=data_dir, csv_file='tiles-v2.csv')
        conversion_skin = {
            'nontumor_skin_necrosis_necrosis':"necrosis",
            'nontumor_skin_muscle_skeletal':        "skeletal muscle",
            'nontumor_skin_sweatglands_sweatglands':        "eccrine sweat glands",
            'nontumor_skin_vessel_vessel':        "vessels",
            'nontumor_skin_elastosis_elastosis':        "elastosis",
            'nontumor_skin_chondraltissue_chondraltissue':        "chondral tissue",
            'nontumor_skin_hairfollicle_hairfollicle':        "hair follicle",
            'nontumor_skin_epidermis_epidermis':        "epidermis",
            'nontumor_skin_nerves_nerves':        "nerves",
            'nontumor_skin_subcutis_subcutis':        "subcutis",
            'nontumor_skin_dermis_dermis':         "dermis",
            'nontumor_skin_sebaceousglands_sebaceousglands':         "sebaceous glands",
            'tumor_skin_epithelial_sqcc':         "squamous-cell carcinoma",
            'tumor_skin_melanoma_melanoma':         "melanoma in-situ",
            'tumor_skin_epithelial_bcc':         "basal-cell carcinoma",
            'tumor_skin_naevus_naevus':        "naevus"
        }
        dataset.classes = [conversion_skin[cl] for cl in dataset.classes]


    elif dataset_name == 'renal':
        dataset = torchvision.datasets.ImageFolder(root=data_dir)
        conversion_nck = {
                    "blood": "red blood cells",
                    "cancer": "renal cancer",
                    "normal": "normal renal tissue",
                    "other": "torn adipose necrotic tissue",
                    "stroma": "muscle fibrous stroma blood vessels",
        }

        dataset.classes = [conversion_nck[cl] for cl in dataset.classes]
        dataset.class_to_idx = {conversion_nck[cl]: idx for cl, idx in dataset.class_to_idx.items()}

    elif dataset_name == "mhist":
        dataset = MhistDataset(root=data_dir, train=False)

    else:
        raise Exception(f'Dataset {dataset_name} not found/implemented!')
    
    
    # Apply the corruption to the dataset transofrms on the fly!
    if corruption == "original":
        print(f"\nNo corruption is applied to the dataset!")

    elif corruption in ['gaussian_noise', 'shot_noise',  'defocus_blur', 'motion_blur',  'brightness', 'contrast' ]:
        if corruption in ['gaussian_noise', 'shot_noise', 'impulse_noise']:
            level = 3
        else:
            level = 5
        original_transforms.transforms.insert(2, CorruptTransform(corruption, level))
        print(f"\nCorruption '{corruption}' is applied to the dataset!")

    elif corruption == "dust":
        original_transforms.transforms.insert(2, AddDust())
        print(f"\nCorruption 'dust' is applied to the dataset!")

    elif corruption == "air_bubble":
        original_transforms.transforms.insert(2, AddAirBubble(min_bubbles=10, max_bubbles=20, transparency=0.3, blur_severity=2))
        print(f"\nCorruption 'dust' is applied to the dataset!")

    elif "stain_light" in corruption:
        original_transforms.transforms.insert(2, Staining(0.05))
        print(f"\nCorruption '{corruption}' is applied to the dataset!")

    elif "stain_heavy" in corruption:
        original_transforms.transforms.insert(2, Staining(0.1))
        print(f"\nCorruption '{corruption}' is applied to the dataset!")
    else:
        raise Exception(f'Corruption {corruption} not found!')

    dataset.transform = original_transforms
    print(f"\n\nFinal Transforms: {dataset.transform}")


    # prepare the data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader, dataset.classes


class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transform=None, train=False, val=False,
                 tumor=False):
        csv_file = os.path.join(root, csv_file)
        self.root = root
        self.data = pd.read_csv(csv_file)

        if train:
            self.data = self.data[self.data['set'] == 'Train']
        else:
            if val:
                self.data = self.data[self.data['set'] == "Validation"]
            else:
                self.data = self.data[self.data['set'] == 'Test']

        if tumor:
            self.data = self.data[self.data['malignicy'] == 'tumor']
        self.tumor = tumor

        self.image_paths = self.data['file'].values
        self.labels = self.data['class'].values

        self.transform = transform
        self.train = train

        self.class_to_idx = {'nontumor_skin_necrosis_necrosis': 0,
                               'nontumor_skin_muscle_skeletal': 1,
                               'nontumor_skin_sweatglands_sweatglands': 2,
                               'nontumor_skin_vessel_vessel': 3,
                               'nontumor_skin_elastosis_elastosis': 4,
                               'nontumor_skin_chondraltissue_chondraltissue': 5,
                               'nontumor_skin_hairfollicle_hairfollicle': 6,
                               'nontumor_skin_epidermis_epidermis': 7,
                               'nontumor_skin_nerves_nerves': 8,
                               'nontumor_skin_subcutis_subcutis': 9,
                               'nontumor_skin_dermis_dermis': 10,
                               'nontumor_skin_sebaceousglands_sebaceousglands': 11,
                               'tumor_skin_epithelial_sqcc': 12,
                               'tumor_skin_melanoma_melanoma': 13,
                               'tumor_skin_epithelial_bcc': 14,
                               'tumor_skin_naevus_naevus': 15
                               }

        self.tumor_map = {'tumor_skin_epithelial_sqcc': 0,
                          'tumor_skin_melanoma_melanoma': 1,
                          'tumor_skin_epithelial_bcc': 2,
                          'tumor_skin_naevus_naevus': 3
                          }

        self.classes = list(self.class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path =  os.path.join(self.root, self.image_paths[index].replace('data/',''))
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if not self.tumor:
            label = self.class_to_idx[self.labels[index]]
        else:
            label = self.tumor_map[self.labels[index]]

        return image, label


class MhistDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file='annotations.csv', image_dir='images', transform=None, train=True):
        csv_file = os.path.join(root, csv_file)
        image_dir = os.path.join(root, image_dir)

        self.data = pd.read_csv(csv_file)
        if train:
            self.data = self.data[self.data['Partition'] == 'train']
        else:
            self.data = self.data[self.data['Partition'] != 'train']
        self.image_paths = self.data['Image Name'].values
        self.labels = self.data['Majority Vote Label'].values
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        self.cat_to_num_map = {'HP': 0, 'SSA': 1}
        self.classes = ["hyperplastic polyp", "sessile serrated adenoma"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.cat_to_num_map[self.labels[index]]

        return image, label
