from constants import defaultkey
import torchvision
import numpy as np

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class FasterRCNN:
    """
    This class is a object detector wrapper class for Fasterrcnn

    Attributes:
    model (detection): pretrained fasterrcnn
    transform (Compose): transformations done to an image such as converting to tensor


    """
    def __init__(self):
        """
        constructor for FasterRCNN class
        """

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        self.model.cuda()
        self.model.eval()
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])

    def get_bboxes(self, frame):
        """
        Given a frame finds all the bounding boxes of people

        Parameters:
        frame (ndarray): frame of a video 

        Returns:
        bboxes_ppl (ndarray): an array of bounding boxes of people
        box_scr (array): array of confidence scores for each bounding box in bboxes_ppl
        """
        img = self.transform(frame)  # Apply the transform to the image
        img = img.cuda()
        pred = self.model([img])  # Pass the image to the model
        pred_dict = pred[0]
        threshold = .5
        bboxes_ppl = [
            bbox for bbox, label, score in zip(pred_dict['boxes'].cuda(
            ), pred_dict['labels'].cuda(), pred_dict['scores'].cuda())
            if COCO_INSTANCE_CATEGORY_NAMES[label] == 'person'
            and score > threshold
        ]
        box_scr = np.array([
            scr.cpu().detach() for scr, label in zip(
                pred_dict['scores'].cuda(), pred_dict['labels'].cuda()) if
            COCO_INSTANCE_CATEGORY_NAMES[label] == 'person' and scr > threshold
        ])
        bboxes_ppl = np.array([[(box[0].cpu().detach(), box[1].cpu().detach()),
                                (box[2].cpu().detach(), box[3].cpu().detach())]
                               for box in bboxes_ppl])
        return bboxes_ppl, box_scr
