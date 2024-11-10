import multiprocessing as mp
import torch
import time
from tqdm import tqdm
from typing import Tuple
from PIL import ImageFont
from fuzzysearch import find_near_matches


from services import yolov10
from services import parseq
from services import craft

# import yolov10
# import parseq
# import craft


MAX_FILE_SIZE = 200 * 1024 * 1024

mp.set_start_method("spawn", force=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load model yolov10 for detect
model_yolov10 = yolov10.load_model_yolov10()

# Load model parseq for recog
img_transform, model = parseq.load_model_parseq()

# load craft model
craft_net, refine_net = craft.load_craft()

# ---------------------
fnt = ImageFont.truetype("Roboto-Medium.ttf", 20)


def result_lp(
    folder_imgs: list, license_plate: str, fuzzy_1: list, fuzzy_2: list
) -> Tuple[bool, str]:
    # start_yolov10 = time.time()
    results = yolov10.detect_yolov10(model_yolov10, folder_imgs)
    # print("Time Yolov10: ", time.time() - start_yolov10)

    license_plate = license_plate.replace("-", "").replace(".", "").upper()

    crop_imgs, bboxes_crops, file_imgs = [], [], []

    # start_parseq = time.time()
    for result in tqdm(results):
        boxes = result.boxes
        file_img = result.path

        if boxes is None:
            continue

        for box in boxes:
            crop_img, bboxes_crop, f_img = parseq.collect_crop(
                file_img, box.xyxy.tolist()[0], craft_net, refine_net
            )
            crop_imgs.append(crop_img)
            bboxes_crops.append(bboxes_crop)
            file_imgs.append(f_img)

    license_plate_detected = parseq.recog_bbox(
        crop_imgs, bboxes_crops, file_imgs, img_transform, model
    )

    # print(f"Time parseq: {time.time() - start_parseq}")

    if isinstance(license_plate_detected, list):
        for lp in license_plate_detected:
            result_fuzzy = find_near_matches(
                license_plate, lp["license_plate"], max_l_dist=2
            )

            if result_fuzzy == []:
                continue
            else:
                # print(f"LP user: {license_plate}, LP detected: {lp['license_plate']}")

                if result_fuzzy[0].dist == 0:
                    print(f"result fuzzy dist = 0: {result_fuzzy[0].dist}")
                    return True, lp["image"]
                elif result_fuzzy[0].dist == 1:
                    print(f"result fuzzy dist = 1: {result_fuzzy[0].dist}")
                    fuzzy_1.append(lp["image"])
                elif result_fuzzy[0].dist == 2:
                    print(f"result fuzzy dist = 2: {result_fuzzy[0].dist}")
                    fuzzy_2.append(lp["image"])

    return False, ""


# if __name__ == "__main__":
#     list_imgs = [
#         "/mlcv2/WorkingSpace/Personal/baotg/TTam/License_Plate/TestSet/0A2A106409534955P9-79322.jpg",
#         "/mlcv2/WorkingSpace/Personal/baotg/TTam/License_Plate/TestSet/FEAD6D5A07494477D1-214.342.jpg",
#         "/mlcv2/WorkingSpace/Personal/baotg/TTam/License_Plate/TestSet/FCF663A107274993H4-61932.jpg",
#         "/mlcv2/WorkingSpace/Personal/baotg/TTam/License_Plate/TestSet/FCD154A007172577F8-99442.jpg",
#         "/mlcv2/WorkingSpace/Personal/baotg/TTam/License_Plate/TestSet/FC7CF09F06543884AF-015162.jpg",
#         "/mlcv2/WorkingSpace/Personal/baotg/TTam/License_Plate/TestSet/0A2A106409534955P9-79322.jpg",
#         "/mlcv2/WorkingSpace/Personal/baotg/TTam/License_Plate/TestSet/FEAD6D5A07494477D1-214.342.jpg",
#         "/mlcv2/WorkingSpace/Personal/baotg/TTam/License_Plate/TestSet/FCF663A107274993H4-61932.jpg",
#         "/mlcv2/WorkingSpace/Personal/baotg/TTam/License_Plate/TestSet/FCD154A007172577F8-99442.jpg",
#         "/mlcv2/WorkingSpace/Personal/baotg/TTam/License_Plate/TestSet/FC7CF09F06543884AF-015162.jpg",
#     ]

#     result_final = result_lp(list_imgs, "84af01516")
#     print(result_final)
