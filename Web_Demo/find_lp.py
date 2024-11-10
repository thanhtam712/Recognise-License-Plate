import argparse
import requests
import math
import time
from PIL import Image
from rich.progress import track
from threading import Thread
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("folder_images", type=str, help="Path folder images")
parser.add_argument("license_plate", type=str, help="Input License Plate")
parser.add_argument(
    "-w", "--number_worker", type=int, default=3, help="Number of worker"
)
args = parser.parse_args()


def format_request(
    list_imgs: list,
    license_plate: str,
    bool_result: list,
    fuzzy_1: list,
    fuzzy_2: list,
    worker: int,
) -> list:
    
    if list_imgs == []:
        bool_result[worker] = False
        return bool_result, fuzzy_1, fuzzy_2
    
    format_request = []

    for img in list_imgs:
        format_request.append(("folder_image", open(img, "rb")))

    if not len(format_request):
        return

    url = "https://aiclub.uit.edu.vn/api/license_plate/items"

    response = requests.post(
        url=url, files=format_request, data={"license_plate": license_plate}
    ).json()
    
    bool_result[worker] = response["result"]
    fuzzy_1[worker] = response["fuzzy_1"]
    fuzzy_2[worker] = response["fuzzy_2"]

    return bool_result, fuzzy_1, fuzzy_2


if __name__ == "__main__":
    folder_images = Path(args.folder_images)
    license_plate = args.license_plate
    workers = args.number_worker

    list_imgs = []
    for p in folder_images.rglob("*"):
        try:
            Image.open(p)
            list_imgs.append(p)
        except Exception:
            print(f"Not an image: {p}")

    check_result = False
    list_fuzzy_1, list_fuzzy_2 = [], []
    start_time = time.time()

    print()
    print("-------------")
    print("Total images:", len(list_imgs))
    print("License plate:", license_plate)
    print("-------------")
    print()

    for st in track(
        range(0, math.ceil(len(list_imgs)), 10 * workers), description="Processing"
    ):
        list_batch = [
            list_imgs[st + 10 * i : st + 10 * (i + 1)] for i in range(workers)
        ]

        list_worker = []
        fuzzy_1, fuzzy_2 = [[] for _ in range(workers)], [[] for _ in range(workers)]
        bool_result = ["" for _ in range(workers)]

        for worker in range(workers):
            t = Thread(
                target=format_request,
                args=(
                    list_batch[worker],
                    license_plate,
                    bool_result,
                    fuzzy_1,
                    fuzzy_2,
                    worker,
                ),
            )
            list_worker.append(t)

        for worker in list_worker:
            worker.start()

        for idx, worker in enumerate(list_worker):
            worker.join()

        for i in range(workers):
            if isinstance(bool_result[i], bool):
                continue
            else:
                print(f"Result: {bool_result[i]}")
                print(f"Time: {time.time() - start_time}")
                check_result = True
                break
            

        if check_result:
            break

        for i in range(workers):
            if fuzzy_1[i] != []:
                for j in range(len(fuzzy_1[i])):
                    list_fuzzy_1.append(fuzzy_1[i][j])
        for i in range(workers):
            if fuzzy_2[i] != []:
                for j in range(len(fuzzy_2[i])):
                    list_fuzzy_2.append(fuzzy_2[i][j])

    if not check_result:
        if list_fuzzy_1 != []:
            print(f"Result with wrong 1 character: {list(list_fuzzy_1[_] for _ in range(len(list_fuzzy_1)))}")
            print(f"Time: {time.time() - start_time}")
        elif list_fuzzy_2 != []:
            print(f"Result with wrong 2 characters: {list(list_fuzzy_2[_] for _ in range(len(list_fuzzy_2)))}")
            print(f"Time: {time.time() - start_time}")
        else:
            print("Not found")
            print(f"Time: {time.time() - start_time}")
