import requests
import os
from zipfile import ZipFile

def get_transcript(path):
    """
    Gets the video transcripts from the video title.
    """
    command = {"b": "bin", "l": "lay", "p": "place", "s": "set"}
    color = {"b": "blue", "g": "green", "r": "red", "w": "white"}
    prep = {"a": "at", "b": "by", "i": "in", "w": "with"}
    digit = { "z": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    adv = { "a": "again", "n": "now", "p": "please", "s": "soon"}

    enc = path.split("/")[-1].split(".")[0]
    dec = []

    for i, c in enumerate(enc):
        if i == 0:
            dec.append(command[c])
        elif i == 1:
            dec.append(color[c])
        elif i == 2:
            dec.append(prep[c])
        elif i == 3:
            dec.append(c)
        elif i == 4:
            dec.append(digit[c])
        elif i == 5:
            dec.append(adv[c])

    return "|".join(dec)

def download_dataset():
    vid_url = "https://spandh.dcs.shef.ac.uk/gridcorpus/s3/video/s3.mpg_vcd.zip"

    r = requests.get(vid_url)

    with open('s3.zip', "wb") as f:
        f.write(r.content)

    with ZipFile("s3.zip", "r") as zip:
        zip.extractall(path = "/data/")


def download_dat():
    url = "https://huggingface.co/public-data/dlib_face_landmark_model/resolve/main/shape_predictor_68_face_landmarks.dat"

    local_filename = "shape_predictor_68_face_landmarks.dat"

    response = requests.get(url, stream=True)

    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)