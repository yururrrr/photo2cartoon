# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from models import ResnetGenerator
import argparse
from utils import Preprocess

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parser = argparse.ArgumentParser()
parser.add_argument('--photo_path', type=str, help='input photo path')
parser.add_argument('--save_path', type=str, help='cartoon save path')
args = parser.parse_args()

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

class Photo2Cartoon:
    def __init__(self):
        self.pre = Preprocess()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
        
        path = 'C:/Users/VIP/Documents/GitHub/photo2cartoon/experiment/train-size256-ch32-True-lr0.0001-adv1-cyc50-id1-identity30-cam1000/cartoon_data_params_latest.pt'
        assert os.path.exists(path), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        params = torch.load(path, map_location=self.device)
        self.net.load_state_dict(params['genA2B'])
        print('[Step1: load ssssweights] success!')

    def inference(self, img):
        # face alignssssssssment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None
        
        print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]
        # post-process
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')
        return cartoon


# if __name__ == '__main__':
#     img = cv2.cvtColor(cv2.imread(args.photo_path), cv2.COLOR_BGR2RGB)
#     c2p = Photo2Cartoon()
#     cartoon = c2p.inference(img)
#     if cartoon is not None:
#         cv2.imwrite(args.save_path, cartoon)
#         print('Cartoon portrait has been saved successfully!')
if __name__ == '__main__':
    c2p = Photo2Cartoon()
    
    input_photos = ['0003.png', '0004.png', '0005.png', '0006.png', '0007.png', '0008.png', '0009.png', '0010.png', '0011.png', '0012.png']  #
    
    input_imgs = []
    output_imgs = []
    
    for photo_path in input_photos:
        path = os.path.join(args.photo_path, photo_path)
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        cartoon = c2p.inference(img)
        if cartoon is not None:
            # cv2.imwrite(args.save_path, cartoon)
            print('Cartoon portrait has been saved successfully!')
            
            img = cv2.resize(img, (256, 256))
            cartoon = cv2.resize(cartoon, (256, 256))
            
         
            input_imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            output_imgs.append(cartoon)
    
    # combine input and output images horizontally
    input_imgs = cv2.hconcat(input_imgs)
    output_imgs = cv2.hconcat(output_imgs)
    
    # combine input and output images vertically
    result = cv2.vconcat([input_imgs, output_imgs])
    
    # save combined image
    cv2.imwrite(args.save_path, result)

