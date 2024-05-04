import os

import numpy as np
import cv2
import torch.utils.data as data


class DUTSDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            # self.image_root = os.path.join(root, "DUTS-TR", "DUTS-TR-Image")
            # self.mask_root = os.path.join(root, "DUTS-TR", "DUTS-TR-Mask")
            self.image_root = os.path.join(root, "train", "imgs")
            self.mask_root = os.path.join(root, "train", "masks")
            self.edge_root = os.path.join(root, "train", "edge_imgs")

        else:
            # self.image_root = os.path.join(root, "DUTS-TE", "DUTS-TE-Image")
            # self.mask_root = os.path.join(root, "DUTS-TE", "DUTS-TE-Mask")
            self.image_root = os.path.join(root, "val", "imgs")
            self.mask_root = os.path.join(root, "val", "masks")
            self.edge_root = os.path.join(root, "val", "edge_imgs")

        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".png")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".png")]
        edge_names = [p for p in os.listdir(self.edge_root) if p.endswith(".png")]

        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check images and mask
        # re_mask_names = []
        # for p in image_names:
        #     mask_name = p.replace(".jpg", ".png")
        #
        #     #根据自己数据集改mask图片名称
        #     # mask_name=mask_name.split('.')[0]+'_json'+'.png'
        #     assert mask_name in mask_names, f"{p} has no corresponding mask."
        #     re_mask_names.append(mask_name)
        # mask_names = re_mask_names

        self.images_path = [os.path.join(self.image_root, n) for n in image_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]
        self.edges_path = [os.path.join(self.edge_root, n) for n in edge_names]


        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        edge_path = self.edges_path[idx]

        #对于原本灰度的图像，进行默认cv::imread读取，读到的仍是CV_8UC3(16)类型的，而非CV_8UC1(1)。其三个通道像素值相等！
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        assert image is not None, f"failed to read image: {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w,_= image.shape

        target = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert target is not None, f"failed to read mask: {mask_path}"

        edge = cv2.imread(edge_path, flags=cv2.IMREAD_GRAYSCALE)
        assert edge is not None, f"failed to read mask: {edge_path}"


        if self.transforms is not None:
            image,target,edge = self.transforms(image,target,edge)

        return image, target,edge

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images, targets,edges = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        batched_edges = cat_list(edges, fill_value=0)

        return batched_imgs, batched_targets,batched_edges


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    train_dataset = DUTSDataset("./data", train=True)
    print(len(train_dataset))

    val_dataset = DUTSDataset("./data", train=False)
    print(len(val_dataset))

    i, t = train_dataset[0]