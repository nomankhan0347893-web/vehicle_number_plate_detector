import cv2
import os
import numpy as np
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def preprocess_plate_for_ocr(plate):

    if plate is None:
        return None

    # upscale only
    plate = cv2.resize(plate, None, fx=5, fy=5, interpolation=cv2.INTER_LANCZOS4)

    # light grayscale
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)

    #fixed blur
    sharpen_kernel=np.array([[-1,-1,-1],
                             [-1,9,-1],
                             [-1,-1,-1]
    ])
    sharpened=cv2.filter2D(denoised,-1,sharpen_kernel)

   #_,thresh=cv2.threshold(
    #    sharpened,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU
    #)


    # mild contrast boost
    #gray = cv2.convertScaleAbs(sharpened, alpha=1.0, beta=10)
    bilateral=cv2.bilateralFilter(sharpened, 9,75,75)

    return cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)

def fix_ocr_errors(text):
    corrections = {
        '0': 'O',
        '|': '',
        '>': '',
        '<': '',
        '*': '',
        ']': '',
        '[': '',
        '(': '',
        ')': '',
        ' ': '',
        '.': '',
        ',': '',
        '_': '',
        '/': '',
        '-': '',
    }
    for old, new in corrections.items():
        text = text.replace(old, new)
    return text

def extract_text_from_plate(plate_image):

    if plate_image is None:
        return ""

    result = reader.readtext(
        plate_image,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
        paragraph=False,
        detail=1
    )

    if not result:
        return ""

    valid_texts = []

    for (bbox, text, confidence) in result:

        # width and height
        width = bbox[1][0] - bbox[0][0]
        height = bbox[2][1] - bbox[1][1]

        area = width * height

        print(text, confidence, area)

        # Ignore tiny detections
        if area < 5000:
            continue

        if confidence > 0.2:

            # x-coordinate for left-to-right sorting
            x = bbox[0][0]

            valid_texts.append((x, text))

    # Sort left to right
    valid_texts = sorted(valid_texts, key=lambda item: item[0])

    # Join texts
    final_text = " ".join([text for (_, text) in valid_texts])
    final_text = fix_ocr_errors(final_text)
    return final_text.upper().strip()


