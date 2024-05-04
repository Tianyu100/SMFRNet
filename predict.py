
import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

# from src import u2net_full
# from MINETcode.network.MINet import MINet_Res50
# from src import F3Net
# from GeleNet.model.GeleNet_models import GeleNet
from SeaNet23.model.SeaNet_models import SeaNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    pre_saved = "./pre_results"
    weights_path = "./save_weights/model_32.pth"  # "./save_weights/best/Zmodel_56(0503).pth"
    img_dirs = "./Tests1"
    imglists = os.listdir(img_dirs)

    threshold = 0.5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),#使用torchvision的transforms，只对输入原图进行压缩
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
#     model = u2net_full() 
#     model = F3Net() 
#     model = GeleNet() 
    model = SeaNet()
    weights = torch.load(weights_path, map_location='cpu')
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)

    t_sum=0
    for imgname in imglists:
        img_path=os.path.join(img_dirs,imgname)
        assert os.path.exists(img_path), f"image file {img_path} dose not exists."

        origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        h, w = origin_img.shape[:2]
        img = data_transform(origin_img)
        img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

        model.eval()
        with torch.no_grad():
        # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

        t_start = time_synchronized()
        #F3Net
#         out1u, out2u, out2r, out3r, out4r, out5r = model(img)
#         pred = out2u
        #GeleNet
#         sal, sal_sig = model(img)
#         pred = sal_sig
        #SeaNet
        res, s34, s5, s12_sig, s34_sig, s5_sig, edge1, edge2 = model(img)
        pred = res.sigmoid()
        
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))
        ts=t_end - t_start
        t_sum=t_sum+ts
        pred = torch.squeeze(pred).to("cpu").detach().numpy()  # [1, 1, H, W] -> [H, W]

        pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        pred_mask = np.where(pred > threshold, 1, 0)
        pred_mask=pred_mask*255.0

        # origin_img = np.array(origin_img, dtype=np.uint8)
        # seg_img = origin_img * pred_mask[..., None]  #在原图上的分割效果


        cv2.imwrite(os.path.join(pre_saved,imgname), cv2.cvtColor(pred_mask.astype(np.uint8), cv2.COLOR_RGB2BGR))

    print("done!")
    print("inference time: {}".format(t_sum/115.0))


if __name__ == '__main__':
    main()
