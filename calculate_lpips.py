import os
import glob
import torch
import lpips
import cv2
from tqdm import tqdm
import numpy as np

loss_fn = lpips.LPIPS(net='alex')

parent_folder = 'result\lpips'

folders = sorted(glob.glob(os.path.join(parent_folder, '*/')))

final = []

for i in range(len(folders)):
    folder1 = folders[i]
    
    for j in range(i+1, len(folders)):
        folder2 = folders[j]
        
        image_files1 = glob.glob(os.path.join(folder1, '*.jpg')) + glob.glob(os.path.join(folder1, '*.png'))
        image_files2 = glob.glob(os.path.join(folder2, '*.jpg')) + glob.glob(os.path.join(folder2, '*.png'))
        
        train_bar = tqdm(range(len(image_files1)))
        lpips_values = []
        for k, _ in enumerate(train_bar):
            image1 = cv2.imread(image_files1[k])[:, :, ::-1]  
            image2 = cv2.imread(image_files2[k])[:, :, ::-1]  
            
            image1 = cv2.imread(image_files1[k])[:, :, ::-1].copy()  
            image2 = cv2.imread(image_files2[k])[:, :, ::-1].copy()  

            image_tensor1 = torch.tensor(image1.transpose((2, 0, 1)), dtype=torch.float32) / 255.0  
            image_tensor2 = torch.tensor(image2.transpose((2, 0, 1)), dtype=torch.float32) / 255.0  

            image_tensor1 = image_tensor1.unsqueeze(0)  
            image_tensor2 = image_tensor2.unsqueeze(0)  

            lpips_value = loss_fn(image_tensor1, image_tensor2)
            train_bar.set_description("F_A: {} | F_B: {} | LPIPS: {: .3f}".format(i, j, lpips_value.item()))

            lpips_values.append(lpips_value.item())
            
        print("LPIPS_val: {}".format(np.mean(np.array(lpips_values))))
        final.append(np.mean(np.array(lpips_values)))

print("Final LPIPS: {}".format(np.array(final)))