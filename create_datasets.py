import torch 
import numpy as np
from torch.utils.data import Dataset

from dataset import DataSet
from load import load_data
from vision_module import feat_rep_vision_module


class ShapesDataset(Dataset):
    """
    This class uses given image, label and feature representation arrays to make a pytorch dataset out of them.
    The feature representations are left empty until 'generate_dataset()' is used to fill them.
    """
    def __init__(self, images=None, labels=None, feat_reps=None, transform=None):
        if images is None and labels is None:
            raise ValueError('No images or labels given')
        self.images = images  # array shape originally [480000,64,64,3], uint8 in range(256)
        self.feat_reps = feat_reps
        self.labels = labels  # array shape originally [480000,6], float64
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def generate_dataset():
    """
    Function to create the feature representations and include them into the dataset
    """
    print("Starting to create the feature representation dataset")
    # load the trained model from save
    model = feat_rep_vision_module()
    model_path = './models/vision_module'
    dataset_path = 'dataset/complete_dataset'
    try:
        model.load_state_dict(torch.load(model_path), strict=False)
    except:
        raise ValueError(f'No trained vision module found in {model_path}. Please train the model first.')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model.to(device)
    model.eval()
    data = torch.load(dataset_path)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=32,
                                              shuffle=False,
                                              pin_memory=True)
    
    images = []
    labels = []
    feature_representations = []

    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            input = input.to(device)
            target = target.to(device=device, dtype=torch.float32)
            
            feat_rep = model(input)

            images_flat = torch.flatten(input, start_dim=0, end_dim=0)
            labels_flat = torch.flatten(target, start_dim=0, end_dim=0)
            feat_rep_flat = torch.flatten(feat_rep, start_dim=0, end_dim=0)

            for image in images_flat:
                images.append(image.cpu().numpy())
            for label in labels_flat:
                labels.append(label.cpu().numpy())
            for feat_rep in feat_rep_flat:
                feature_representations.append(feat_rep.cpu().numpy())

    # for size reasons the dataset is saved twice, 
    # once as the full dataset now including the feature representations
    feat_rep_dataset_full = ShapesDataset(np.array(images), np.array(labels), np.array(feature_representations))
    torch.save(feat_rep_dataset_full, dataset_path + '_feat_rep')

    # and once as a much smaller dataset with the labels and feature representations but without the original images
    feat_rep_dataset_without_images = ShapesDataset(labels=np.array(labels), feat_reps=np.array(feature_representations))
    torch.save(feat_rep_dataset_without_images, dataset_path + '_feat_rep_no_images')

    print(f"Feature representations saved to {dataset_path + '_feat_rep'} and {dataset_path + '_feat_rep_no_images'}")
    return feat_rep_dataset_full, feat_rep_dataset_without_images

if __name__ == "__main__":
    feat_rep_dataset_path = 'dataset/complete_dataset_20241027_norm' + '_feat_rep'
    try:
        complete_dataset = torch.load(feat_rep_dataset_path)
    except:
        print('Feature representations not found, creating it instead...')
        complete_dataset, _ = generate_dataset()

    # generate concept datasets for the communication game
    feat_rep_concept_dataset = DataSet(game_size=4, is_shapes3d=True, images=complete_dataset.feat_reps, labels=complete_dataset.labels, device='mps')
    torch.save(feat_rep_concept_dataset, './dataset/feat_rep_concept_dataset_new')

    # also for the zero_shot dataset
    feat_rep_zero_concept_dataset = DataSet(game_size=4, zero_shot=True, is_shapes3d=True, images=complete_dataset.feat_reps, labels=complete_dataset.labels, device='mps')
    torch.save(feat_rep_zero_concept_dataset, './dataset/feat_rep_zero_concept_dataset_new')
