import os
import shutil
import string
import random
from typing import List

from PIL import Image
from fastapi import UploadFile


def convert_file(folder_image: List[UploadFile]) -> list:
    if folder_image is None:
        return []

    list_imgs = []

    for img_byte in folder_image:
        path_file = str(img_byte.filename)
        with open(path_file, "wb") as buffer:
            shutil.copyfileobj(img_byte.file, buffer)
        list_imgs.append(path_file)
    return list_imgs


def remove_files(list_imgs: list):
    for file_img in list_imgs:
        Image.open(file_img).close()
        os.remove(file_img)
