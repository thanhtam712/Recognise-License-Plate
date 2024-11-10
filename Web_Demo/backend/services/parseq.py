import cv2
import sys
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from services.craft import detect_text, load_craft

sys.path.append("/mlcv2/WorkingSpace/Personal/baotg/HCMAI24/parseq")
from strhub.data.module import SceneTextDataModule


fnt = ImageFont.truetype("Roboto-Medium.ttf", 20)


def calcul_area(box):
    x1, y1, x2, y2 = box
    return abs(x1 - x2) * abs(y1 - y2)


def calcul_iou(box_lhs, box_rhs, thresh_iou=0.5):
    x1_lhs, y1_lhs, x2_lhs, y2_lhs = box_lhs
    x1_rhs, y1_rhs, x2_rhs, y2_rhs = box_rhs

    area_lhs = calcul_area(box_lhs)
    area_rhs = calcul_area(box_rhs)

    # Determines the coordinates of the intersection box
    x1_inter = max(x1_lhs, x1_rhs)
    y1_inter = max(y1_lhs, y1_rhs)
    x2_inter = min(x2_lhs, x2_rhs)
    y2_inter = min(y2_lhs, y2_rhs)

    # Determines if the boxes overlap or not
    # If one of the two is equal to 0, the boxes do not overlap
    inter_w = max(0.0, x2_inter - x1_inter)
    inter_h = max(0.0, y2_inter - y1_inter)

    if inter_w == 0.0 or inter_h == 0.0:
        return 0.0

    intersection_area = inter_w * inter_h
    union_area = area_lhs + area_rhs - intersection_area

    iou: float = intersection_area / union_area

    return iou


def load_model_parseq():
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True).eval()
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    return img_transform, model


def collect_crop(file_img: str, bounding_box: list, craft_net, refine_net):
    x_min, y_min, x_max, y_max = bounding_box
    image_ori = Image.open(file_img)

    img_visualize = ImageDraw.Draw(image_ori)
    image = image_ori.crop((x_min, y_min, x_max, y_max))
    image = np.asarray(image)

    image_numpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_numpy)

    max_bboxes = []

    # start_craft = time.time()
    for i in range(10):
        prediction_result = detect_text(image_numpy, craft_net, refine_net, i)

        if len(prediction_result["boxes"]) == 2:
            if not len(max_bboxes):
                for box in prediction_result["boxes"]:
                    rect_bbox = [
                        np.min(box[:, 0]),
                        np.min(box[:, 1]),
                        np.max(box[:, 0]),
                        np.max(box[:, 1]),
                    ]

                    if (
                        calcul_iou(rect_bbox, [0, 0, image.width, image.height]) > 0.7
                        or (rect_bbox[3] - rect_bbox[1]) > 0.6 * image.height
                        or (rect_bbox[2] - rect_bbox[0]) < 0.6 * image.width
                    ):
                        max_bboxes = []
                        break

                    max_bboxes.append(rect_bbox)

                if (
                    len(max_bboxes) == 2
                    and calcul_iou(max_bboxes[0], max_bboxes[1]) > 0.5
                ):
                    max_bboxes = []
            else:
                backup_bboxes = max_bboxes[:]

                for i, box in enumerate(prediction_result["boxes"]):
                    rect_bbox = [
                        np.min(box[:, 0]),
                        np.min(box[:, 1]),
                        np.max(box[:, 0]),
                        np.max(box[:, 1]),
                    ]

                    if (
                        calcul_iou(rect_bbox, [0, 0, image.width, image.height]) > 0.7
                        or (rect_bbox[3] - rect_bbox[1]) > 0.6 * image.height
                        or (rect_bbox[2] - rect_bbox[0]) < 0.6 * image.width
                    ):
                        max_bboxes = backup_bboxes[:]
                        break

                    if calcul_area(rect_bbox) > calcul_area(max_bboxes[i]):
                        max_bboxes[i] = rect_bbox

                if (
                    len(max_bboxes) == 2
                    and calcul_iou(max_bboxes[0], max_bboxes[1]) > 0.5
                ):
                    max_bboxes = backup_bboxes[:]
        else:
            pass

    img_visualize.rectangle(bounding_box, outline="red", width=2)

    if not len(max_bboxes):
        # if no any bbox text detected --> CRAFT detected two lines as one line
        # then we pseudo split image into 2 parts, with overlap 0.1
        max_bboxes = [
            [0, 0, image.width, image.height // 2 + 0.1 * image.height],
            [0, image.height // 2 - 0.1 * image.height, image.width, image.height],
        ]

    # print(f"Time CRAFT: {time.time() - start_craft}")

    bboxes = sorted(max_bboxes, key=lambda x: x[1])

    if not bboxes:
        return

    crops = [
        image.crop((x_min, y_min, x_max, y_max))
        for (x_min, y_min, x_max, y_max) in bboxes
    ]
    return crops, bboxes, file_img


def recog_bbox(crops: list, bboxes: list, file_imgs: list, img_transform, model):
    license_plates = []

    for idx, (crop_img, bbox, file_img) in enumerate(zip(crops, bboxes, file_imgs)):
        all_images = torch.stack([img_transform(crop) for crop in crop_img], dim=0)

        with torch.no_grad():
            logits = model(all_images)

        pred = logits.softmax(-1)
        labels, confidences = model.tokenizer.decode(pred)
        texts_imgs = []

        for _idx, box in enumerate(bbox):
            _text = {
                "text": str(labels[_idx]),
                "bbox": box,
            }
            texts_imgs.append(_text)

        license_plate = []
        for text in sorted(texts_imgs, key=lambda x: x["bbox"][1]):
            for character in text["text"]:
                if character.isalpha() or character.isdigit():
                    license_plate.append("".join(character))

        context = {
            "image": file_img,
            "license_plate": "".join(license_plate),
        }

        license_plates.append(context)
        print(license_plates)
    return license_plates
