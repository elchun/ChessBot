# Training script so I can run multiprocess
# Run this from the src directory

# Download TorchVision repo to use some files from
# references/detection
# !git clone https://github.com/pytorch/vision.git
# !cd vision && git checkout v0.3.0
# !cp vision/references/detection/utils.py ./
# !cp vision/references/detection/transforms.py ./
# !cp vision/references/detection/coco_eval.py ./
# !cp vision/references/detection/engine.py ./
# !cp vision/references/detection/coco_utils.py ./

# Imports
import os.path as osp

import fnmatch
import json
import matplotlib.pyplot as plt
# import multiprocessing
import numpy as np
import os
from PIL import Image
from IPython.display import display

import torch
import torch.utils.data

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

from engine import train_one_epoch, evaluate
import vision.references.detection.utils as utils
import vision.references.detection.transforms as T
# import transforms as T

# -- Consts -- #

dataset_path = 'temp_data'

ycb = [
    "003_cracker_box.sdf", "004_sugar_box.sdf", "005_tomato_soup_can.sdf",
    "006_mustard_bottle.sdf", "009_gelatin_box.sdf", "010_potted_meat_can.sdf"
]

pieces = [
    'BB', # : 'Bishop_B.urdf',
    'BW', # : 'Bishop_W.urdf',

    'KB', # : 'King_B.urdf',
    'KW', # : 'King_W.urdf',

    'NB', # : 'Knight_B.urdf',
    'NW', # : 'Knight_W.urdf',

    'PB', # : 'Pawn_B.urdf',
    'PW', # : 'Pawn_W.urdf',

    'QB', # : 'Queen_B.urdf',
    'QW', # : 'Queen_W.urdf',

    'RB', # : 'Rook_B.urdf',
    'RW', # : 'Rook_W.urdf'
]


# -- Define dataloader -- #
class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.num_images = len(fnmatch.filter(os.listdir(root),'*.png'))

        with open(osp.join(root, 'id_to_name.json'), "r") as f:
            instance_id_to_class_name = json.load(f)

        self.instance_id_to_class_name = instance_id_to_class_name
        self.obj_ids = np.asarray(list(instance_id_to_class_name.keys()))

    def __getitem__(self, idx):
        filename_base = os.path.join(self.root, f"board_{idx:05}")

        img = Image.open(filename_base + '.png').convert('RGB')
        mask = np.squeeze(np.load(filename_base + '_mask.npy'))

        count = (mask == np.int16(self.obj_ids)[:, None, None]).sum(axis=2).sum(axis=1)

        # print(count)

        # discard objects instances with less than 10 pixels
        local_obj_ids = self.obj_ids[count >= 10]

        labels = []
        for id in local_obj_ids:
            piece = self.instance_id_to_class_name[id][-2:]  # piece type is last two values
            labels.append(pieces.index(piece))

        local_obj_ids = np.int16(np.asarray(local_obj_ids))

        masks = mask == local_obj_ids[:, None, None]  # bool mask
        # print('masks: ', masks.shape)

        # get bounding box coordinates for each mask
        num_objs = len(local_obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.num_images


# -- Make network -- #
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

# -- Get transform -- #

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == '__main__':
    # use our dataset and defined transformations
    dataset = ChessDataset(dataset_path, get_transform(train=True))
    dataset_test = ChessDataset(dataset_path, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # Multiprocessing doesn't work in ipynb on mac... might work with cuda tho
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=8,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=8,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = len(pieces)+1

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        torch.save(model.state_dict(), 'chess_maskrcnn_model_ep_{epoch:01}.pt')

    torch.save(model.state_dict(), 'chess_maskrcnn_model.pt')