import numpy as np
import torch.nn.functional as F
import torch
def diff_pixel(video,threshold=0.1,patch_size=14,stride = 2):

    pixel_diff = video[1:] - video[:-1]
    pixel_diff = torch.abs(pixel_diff)
    patches_diff = F.avg_pool2d(pixel_diff,(patch_size,patch_size))     # 按照patches来划分
    patches_diff = F.avg_pool2d(patches_diff, stride)                   # llava-video会有一个merge操作
    t,channel,patches_height,patches_weight = patches_diff.shape
    patches_diff = patches_diff.view(t, channel , patches_height * patches_weight)
    patches_diff = patches_diff.mean(dim=1)         # 在channel维度上进行mean值
    # Create a dummy first frame for each item in the batch
    first_frame = torch.ones_like(patches_diff[0:1]) * 255
    patches_diff = torch.cat((first_frame, patches_diff), dim=0)
    
    mask = patches_diff > threshold

    mask = mask.view(t+1,patches_height,patches_weight)        # [64,169] -> [64,13,13]
    n_mask = torch.ones(t+1, patches_height, 1, dtype=torch.bool).to(mask.device)     # 换行符设置为true
    mask = torch.cat((mask,n_mask),dim=-1)
    mask = mask.view(-1)
    return mask
def divide_scene_pixel_level(video,threshold=0.1,scene_change_threshold=0.3,patch_size=14,stride = 2):
    dummy_frame = torch.ones_like(video[0:1]) * 255
    video = torch.cat((dummy_frame, video), dim=0)
    pixel_diff = video[1:] - video[:-1]
    pixel_diff = torch.abs(pixel_diff)
    patches_diff = F.avg_pool2d(pixel_diff,(patch_size,patch_size))     # 按照patches来划分
    patches_diff = F.avg_pool2d(patches_diff, stride)                   # llava-video会有一个merge操作
    patches_diff = patches_diff.mean(dim=1)
    mask = patches_diff>threshold
    each_frame_pixel_change_num = mask.sum(dim=(1,2))

    t,height,weight = patches_diff.shape
    each_frame_pixel_change_ratio = each_frame_pixel_change_num/(height*weight)
    scene_change_idx = torch.where(each_frame_pixel_change_ratio>scene_change_threshold)[0]
    print(scene_change_idx)
    return scene_change_idx

def divide_scene_patch_level(video,threshold=0.1,scene_change_threshold=0.3,patch_size=14,stride = 2):
    dummy_frame = torch.ones_like(video[0:1]) * 255
    video = torch.cat((dummy_frame, video), dim=0)
    pixel_diff = video[1:] - video[:-1]
    pixel_diff = torch.abs(pixel_diff)
    patches_diff = F.avg_pool2d(pixel_diff,(patch_size,patch_size))     # 按照patches来划分
    patches_diff = F.avg_pool2d(patches_diff, stride)                   # llava-video会有一个merge操作
    patches_diff = patches_diff.mean(dim=1)
    mask = patches_diff>threshold
    each_frame_pixel_change_num = mask.sum(dim=(1,2))

    t,height,weight = patches_diff.shape
    each_frame_pixel_change_ratio = each_frame_pixel_change_num/(height*weight)
    scene_change_idx = torch.where(each_frame_pixel_change_ratio>scene_change_threshold)[0]
    return scene_change_idx
