from einops import rearrange
import numpy as np
import torch
from torchvision import transforms

# Local imports
from datasets.kitti import KITTI

# Deprecated - norm not needed.
################## ORB.py ################## 
# Used to generate means and stds for keypoints. 
# Means and std values are hard-coded directly into kitti.py

OUT_PATH = './keypoint_norm_params.txt'

def get_norm_params_keypoints(dataloader):
    x_values = []
    y_values = []
    size_values = []
    angle_values = []
    for images, keypoints, gt in dataloader:
        keypoints = rearrange(keypoints, 'b t k v -> (b t k) v').numpy()
        x_values.extend(keypoints[:, 0])
        y_values.extend(keypoints[:, 1])
        size_values.extend(keypoints[:, 2])
        angle_values.extend(keypoints[:, 3])

    # convert angle to rad
    angle_values = np.radians(angle_values)

    # calculate mean
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    size_mean = np.mean(size_values)
    angle_mean = np.mean(angle_values)

    # calculate std
    x_std = np.std(x_values)
    y_std = np.std(y_values)
    size_std = np.std(size_values)
    angle_std = np.std(angle_values)

    # print to .txt file
    with open(OUT_PATH, 'w') as f:
        f.write("mean values (x, y, size, angle):\n")
        f.write(f"[{x_mean}, {y_mean}, {size_mean}, {angle_mean}]\n")
        f.write("std values (x, y, size, angle):\n")
        f.write(f"[{x_std}, {y_std}, {size_std}, {angle_std}]\n")





if __name__ == '__main__':
    resize = transforms.Resize(((192, 640)))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.34721234, 0.36705238, 0.36066107],
            std=[0.30737526, 0.31515116, 0.32020183]),
    ])
    dataset_params = {
        "window_size": 2,
        "overlap": 1,
        "bsize": 4
    }

    dataset = KITTI(window_size=dataset_params["window_size"], overlap=dataset_params["overlap"], resize_transform=resize, preprocess_transform=preprocess, normalize_keypoints=False)

    dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=dataset_params["bsize"],
                                               shuffle=True,
                                               )
    
    get_norm_params_keypoints(dataloader)    