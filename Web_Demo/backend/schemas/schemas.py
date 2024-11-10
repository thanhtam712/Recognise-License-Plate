from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List


class SearchLP(BaseModel):
    license_plate: str
    folder_image: List[UploadFile] = File(...)
