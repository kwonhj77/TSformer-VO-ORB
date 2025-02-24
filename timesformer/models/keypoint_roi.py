from einops import rearrange
import numpy as np
import torch
from torch import nn
from torchvision.ops import roi_align

class KeypointPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, image_size=(192, 640), patch_size=16, num_keypoints=25, embed_dim=768): # H, W
        super().__init__()
        self.in_channels = in_channels
        self.aligned = True
        self.spatial_scale = 1.0
        self.sampling_ratio = -1
        self.image_size = image_size
        self.output_size = (patch_size, patch_size) # Ph, Pw
        self.num_keypoints = num_keypoints
        self.embed_dim = embed_dim
        self.embed_patches = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, padding=0)

    def convert_pt_r_to_xyxy(self, keypoint):
        x, y, r = keypoint[0], keypoint[1], keypoint[2]
        x1 = max(x-r, 0)
        y1 = max(y-r, 0)
        x2 = min(x+r, self.image_size[1])
        y2 = min(x+r, self.image_size[0])
        return torch.tensor([x1, y1, x2, y2])

    def forward(self, features, keypoints):
        B, C, T, H, W = features.shape
        _, _, O, _ = keypoints.shape # B, T, O, 3
        assert B == keypoints.shape[0]
        assert C == self.in_channels
        assert T == keypoints.shape[1]
        assert H == self.image_size[0]
        assert W == self.image_size[1]
        assert O == self.num_keypoints

        features = rearrange(features, 'b c t h w -> (b t) c h w')
        keypoints = rearrange(keypoints, 'b t o k -> (b t o) k')

        keypoints_xyxy = keypoints.new_tensor(np.zeros((keypoints.shape[0], 4)))

        for i in range(keypoints.shape[0]):
            keypoints_xyxy[i,:] = self.convert_pt_r_to_xyxy(keypoints[i,:])

        keypoints_xyxy = rearrange(keypoints_xyxy, '(b t o) k -> (b t) o k', b=B, t=T, o=O)
        
        ret = roi_align(
            input=features,
            boxes=list(keypoints_xyxy.float()),
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned
        ) # BTO, C, Ph, Pw

        ret_embedded = self.embed_patches(ret).flatten(1) # BTO, d

        ret_embedded = rearrange(ret_embedded, '(b t o) d -> b t o d', b=B, t=T, o=O).contiguous() # B, T, O, d
        return ret_embedded




