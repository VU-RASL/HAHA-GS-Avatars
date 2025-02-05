import os
import pickle
from glob import glob

import cv2
import numpy as np
import torch
import smplx 

def path_to_id(path):
    filename = path.split('/')[-1]
    no_extension = filename.split('.')[0]
    return no_extension


class DataLoader(torch.utils.data.Dataset):
    """
    Use fitted SMPL-X model as input data.
    Folder with the dataset should contain:
     * images - Folder with input images *****.png
     * masks - Folder with segmentation masks made by Graphonamy: *****.png
     * smplx - Fitting results of SMPLify-X
    """
    def __init__(
            self,
            data_root,
            use_hashing=False,
            random_background=False,
            white_background=False,
            render_size=640,
    ):
        self._images_path = os.path.join(data_root, "images")
        self._masks_path = os.path.join(data_root, "masks")
        self._smplx_path = os.path.join(data_root, "smplx/results")
        self._render_size = render_size

        # Get images list
        self._images_list = glob(os.path.join(self._images_path, "*"))
        self._images_list = sorted(self._images_list, key=lambda x: int(path_to_id(x)))
        self._len = len(self._images_list)

        self._random_background = random_background
        self._white_background = white_background
        self._use_hashing = use_hashing
        self._data_hash = {}

        self.sequence_name = data_root.split('/')[-1].split('-test')[0]
        

    def load_sample(self, pid):
        #print("Im here" )
        #print("Im here" )

        rgb_image = cv2.imread(os.path.join(self._images_path, pid + ".png"))[..., ::-1]
        rgb_image = rgb_image.astype(np.float32) / 255.0
        rgb_image = cv2.resize(rgb_image, (self._render_size, self._render_size))
        rgb_image = np.transpose(rgb_image, (2, 0, 1))

        if self._white_background:
            background = np.ones_like(rgb_image)
        else:
            background = np.zeros_like(rgb_image)

        if self._random_background:
            background = np.ones_like(rgb_image)
            background_color = np.random.rand(3).astype(np.float32)
            background[0] *= background_color[0]
            background[1] *= background_color[1]
            background[2] *= background_color[2]

        mask_image = cv2.imread(os.path.join(self._masks_path, pid + ".png"))
        #print(mask_image.shape)
        skin_mask = np.zeros_like(mask_image).astype(int)
        for label in [10, 13, 14, 15, 16, 17]:  # All skin labels
            skin_mask |= (mask_image == label).astype(int)
        skin_mask = skin_mask.astype(np.float32)
        skin_mask = skin_mask[None, :, :, 0]

        mask_image = mask_image.astype(np.float32)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        mask_image = cv2.resize(mask_image, (self._render_size, self._render_size))
        _, mask_image = cv2.threshold(mask_image, 0.01, 1, cv2.THRESH_BINARY)

        mask_image = mask_image[None]
        rgb_image = mask_image * rgb_image + (1 - mask_image) * background

        
        # Load pickle
        smplx_filename = os.path.join(self._smplx_path, pid, "000.pkl")

        with open(smplx_filename, "rb") as f:
            smplx_params = pickle.load(f)

        import math
        
        def focal2fov(focal, pixels):
            return 2 * math.atan(pixels / (2 * focal))

        def getProjectionMatrix(znear, zfar, fovX, fovY):
            tanHalfFovY = math.tan((fovY / 2))
            tanHalfFovX = math.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            P = np.zeros((4, 4))

            z_sign = 1.0

            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P


        def K_to_camera_matrix(K, src_W, src_H):
            scale_y = src_H / src_W
            FoVx = focal2fov(K[0, 0], src_W)
            FoVy = focal2fov(K[1, 1], src_H) / scale_y
            camera_matrix = getProjectionMatrix(znear=0.01, zfar=1.0, fovX=FoVx, fovY=FoVy)
            camera_matrix[2, 2] = 1.0001
            return camera_matrix

        #print(pid,smplx_filename)
        for key in smplx_params.keys():
            if key =='betas':
                #print('111')
                smplx_params[key] = smplx_params[key][:,:20]
            if key =='camera_translation':
                camera_translation = torch.tensor(smplx_params['camera_translation'])
            if key =='camera_rotation':
                camera_rotation= torch.tensor(smplx_params['camera_rotation'])
            
            #print(key, smplx_params[key].shape)
        intrinsic = torch.tensor([
                    [1000, 0, 810/2],
                    [0, 650, 1200/2],
                    [0, 0, 1]
                ]) 
            
        a = camera_translation
        camera_translation = camera_translation.unsqueeze(-1)  # Shape: (1, 3, 1)
    
        # Concatenate rotation and translation to form [R | t]
        extrinsic = torch.cat([camera_rotation, camera_translation], dim=2)    # Shape: (1, 3, 4)

        smplx_params["camera_matrix"] = K_to_camera_matrix(intrinsic,900/2, 756/2)
        smplx_params["camera_transform"] = torch.tensor(np.array([   [1,0,0,0],
                                                        [0,1,0,0],
                                                        [0,0,1,0],
                                                        [0,0,0,1]
                                                        ]))
        #print("camera_transform ______________ ",smplx_params["camera_transform"].shape)
        smplx_params["camera_matrix"] = smplx_params["camera_matrix"].astype(np.float32)
        smplx_params["camera_transform"] = smplx_params["camera_transform"].numpy().astype(np.float32)
        smplx_params['transl'] = a
        

        return {
            "pid": pid,
            "rgb_image": rgb_image,
            "mask_image": mask_image,
            "skin_mask": skin_mask,
            "background": background,
            "smplx_params": smplx_params,
        }

    def __getitem__(self, index):
        index = index % self._len
        rgb_filename = self._images_list[index]
        pid = rgb_filename.split("/")[-1].split(".")[0]

        if rgb_filename in self._data_hash:
            data_dict = self._data_hash[rgb_filename]
        else:
            data_dict = self.load_sample(pid)
            if self._use_hashing:
                self._data_hash[rgb_filename] = data_dict
        return data_dict

    def __len__(self):
        return self._len
