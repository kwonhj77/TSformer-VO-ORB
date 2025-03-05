import cv2
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

# Local imports
from datasets.kitti import KITTI




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

    dataset = KITTI(window_size=dataset_params["window_size"], overlap=dataset_params["overlap"], resize_transform=resize, preprocess_transform=preprocess, use_keypoints=True, num_keypoints=200, use_descriptors=True)

    dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=dataset_params["bsize"],
                                               shuffle=True,
                                               )
    all_keypoint_sizes = []
    ctr = 0
    for batch_images, batch_keypoints, batch_gt, batch_descriptors in dataloader:

        ###  Messing around with the matcher  ###
        # matcher = cv2.BFMatcher()

        # imgs = rearrange(batch_images[0], 'c t h w -> t c h w').numpy()
        # train_img = imgs[0]
        # query_img = imgs[1]

        # keypoints = batch_keypoints[0].numpy()
        # train_keypoint = keypoints[0]
        # query_keypoint = keypoints[1]


        # descriptors = batch_descriptors[0].numpy() # Get first item in batch
        # train_descriptor = descriptors[0]
        # query_descriptor = descriptors[1]

        # matches = matcher.match(query_descriptor, train_descriptor)

        # final_img = cv2.drawMatches(query_img, query_keypoint, train_img, train_keypoint, matches[:20], None)

        # cv2.imshow("plot1", final_img)        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        ### Understanding keypoint sizes ###
        keypoints = batch_keypoints.numpy()[:,0,:,:] # B, T, num_keypoints, (x y size)
        keypoint_sizes = keypoints.reshape(keypoints.shape[0]*keypoints.shape[1], keypoints.shape[2])[:,2]
        all_keypoint_sizes.extend(list(keypoint_sizes))

        ctr += 1
        # if ctr > 10000:
        #     break
    
    plt.hist(all_keypoint_sizes, bins=30)
    plt.xlabel('keypoint sizes')
    plt.ylabel('freq')
    plt.title('keypoint sizes')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()


        
    print("Exiting...")