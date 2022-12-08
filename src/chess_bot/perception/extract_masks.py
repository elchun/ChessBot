import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from PIL import Image
import os.path as osp
import io

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

        self.DeclareAbstractOutputPort(
            'raw_prediction',
            lambda: AbstractValue.Make([torch.Tensor, np.ndarray]),
            self.get_raw_prediction)

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
        self.prev_prediction = None

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

        self.prev_prediction = prediction

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

    def get_raw_prediction(self, context, output):
        img = self.GetInputPort('rgb_image').Eval(context).data
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        # Only recalculate prediction when getting masks
        if self.prev_prediction is None:
            with torch.no_grad():
                prediction = self.model([Tf.to_tensor(img).to(self.device)])
        else:
            prediction = self.prev_prediction
        output.set_value((prediction, img))
        # thresh = 0.97
        # img_np = np.array(img)
        # fig, ax = plt.subplots(1, figsize=(12,9))
        # ax.imshow(img_np)

        # cmap = plt.get_cmap('tab20b')
        # colors = [cmap(i) for i in np.linspace(0, 1, 60)]

        # num_instances = prediction[0]['boxes'].shape[0]
        # bbox_colors = random.sample(colors, num_instances)
        # boxes = prediction[0]['boxes'].cpu().numpy()
        # labels = prediction[0]['labels'].cpu().numpy()
        # scores = prediction[0]['scores'].cpu().detach().numpy()


        # for i in range(num_instances):
        #     if scores[i] < thresh:
        #         continue
        #     color = bbox_colors[i]
        #     bb = boxes[i,:]
        #     bbox = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
        #             linewidth=2, edgecolor=color, facecolor='none')
        #     ax.add_patch(bbox)
        #     plt.text(bb[0], bb[1], s=self.get_piece_from_label(labels[i]),
        #             color='white', verticalalignment='top',
        #             bbox={'color': color, 'pad': 0})
        #     plt.text(bb[0], bb[3], s=str(f'{scores[i]:.3}'),
        #             color='white', verticalalignment='bottom',
        #             bbox={'color': color, 'pad': 0})

        # plt.axis('off');
        # plt.show()
        # fig.canvas.draw()

        # fig.savefig('/tmp/chess_masks.png')

        # # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        # io_buf = io.BytesIO()
        # fig.savefig(io_buf, format='raw')
        # io_buf.seek(0)
        # img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        #                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        # io_buf.close()

        # output.set_value('/tmp/chess_masks.png')