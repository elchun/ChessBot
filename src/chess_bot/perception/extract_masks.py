import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os.path as osp

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as Tf


from pydrake.all import (
    AbstractValue,
    LeafSystem)

from chess_bot.utils.path_util import get_chessbot_src

class ExtractMasks(LeafSystem):
    """
    System to extract masks from rgbd image using Mask R-CNN.
    """
    def __init__(self, rgbd_sensor):
        LeafSystem.__init__(self)
        self.DeclareAbstractInputPort(
            'rgb_image',
            rgbd_sensor.color_image_output_port().Allocate())
        self.DeclareAbstractInputPort(
            'depth_image',
            rgbd_sensor.depth_image_32F_output_port().Allocate())

        self.DeclareAbstractOutputPort(
            'masked_depth_image',
            lambda: AbstractValue.Make([np.ndarray]),
            self.get_masks)

        self.pieces = [
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

        # Higher angle and random orientation
        model_file = osp.join(get_chessbot_src(), 'resources/weights/chess_maskrcnn_model_ror.pt')
        # model_file = 'weights/chess_maskrcnn_model_ror.pt'  # Higher angle and random orient

        # Set up model
        self.model = self.get_instance_segmentation_model(len(self.pieces) + 1)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
            'cpu')
        self.model.load_state_dict(
            torch.load(model_file, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def get_piece_from_label(self, label):
        return self.pieces[label - 1]


    def get_instance_segmentation_model(self, num_classes):
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

    def get_masks(self, context, output):
        thresh = 0.97
        # print(self.EvalAbstractInput(context, 'rgb_image').get_value())
        # res = np.repeat(rgb_image[np.newaxis, :, :], 5, axis=0)

        color_image = self.GetInputPort('rgb_image').Eval(context).data
        color_image = Image.fromarray(np.uint8(color_image)).convert('RGB')
        depth_image = self.GetInputPort('depth_image').Eval(context).data
        depth_image = depth_image.squeeze()

        with torch.no_grad():
            prediction = self.model([Tf.to_tensor(color_image).to(self.device)])

        labels = list(prediction[0]['labels'].cpu().detach().numpy())
        scores = list(prediction[0]['scores'].cpu().detach().numpy())
        masks = prediction[0]['masks'].cpu().detach().numpy()

        masked_depth_imgs = []
        predicted_pieces = []
        for i, mask in enumerate(masks):
            if scores[i] < thresh:
                continue
            mask = mask.squeeze()
            mask = mask > 0.7  # Threshold mask so it only uses high confidence
            masked_depth_img = depth_image * mask
            masked_depth_imgs.append(masked_depth_img)
            predicted_pieces.append(self.get_piece_from_label(labels[i]))

        output.set_value([np.stack(masked_depth_imgs, axis=0), predicted_pieces])