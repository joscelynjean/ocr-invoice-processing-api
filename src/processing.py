import pytesseract
import pandas as pd
import re

def pre_processing_image(img):
    return img

def text_recognition_from_image(img):
    return pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

def build_dataframe_from_ocr_data(ocr_data):
    df = pd.DataFrame(ocr_data)
    
    df['match_qst_pattern'] = df.text.apply(lambda t: int(bool(re.search("\d{10}TQ\d{4}", t, re.IGNORECASE))))
    df['match_gst_pattern'] = df.text.apply(lambda t: int(bool(re.search("\d{9}", t, re.IGNORECASE))))
    df['amount'] = df.text.apply(lambda t: int(bool(re.search("^[+-]?[0-9]{1,3}(?:,?[0-9]{3})*[\.|,][0-9]{2} ?\$?$", t))))
    
    return df