from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import Body
from typing import List
from pathlib import Path

from schemas.schemas import SearchLP
from services.pipeline import result_lp
from utils.save_imgs import convert_file, remove_files


router = APIRouter()


@router.post("/items/")
async def recog_img(
    license_plate: str = Body(...), folder_image: List[UploadFile] = File(None)
):
    fuzzy_1, fuzzy_2 = [], []
    list_imgs = convert_file(folder_image)
    result, f_img = result_lp(
        list_imgs, license_plate, fuzzy_1=fuzzy_1, fuzzy_2=fuzzy_2
    )

    remove_files(list_imgs)

    if f_img != "" and str(f_img).split(".")[-1] in ["jpg", "png"]:
        print(f"Result with fuzzy 0: {f_img}")
        return JSONResponse(
        content={"result": Path(f_img).name, "fuzzy_1": fuzzy_1, "fuzzy_2": fuzzy_2}
        )

    if fuzzy_1 != []:
        print(f"Result with fuzzy 1: {fuzzy_1}")
    elif fuzzy_2 != []:
        print(f"Result with fuzzy 2: {fuzzy_2}")


    return JSONResponse(
        content={"result": result, "fuzzy_1": fuzzy_1, "fuzzy_2": fuzzy_2}
    )
