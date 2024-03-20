import pytesseract
import pandas as pd
import re
import numpy as np

from utils import read_csv_to_array

gst_amount_labels = read_csv_to_array('data/keywords/gstAmount.csv')
gst_number_labels = read_csv_to_array('data/keywords/gstNumber.csv')
invoice_number_labels = read_csv_to_array('data/keywords/invoiceNumber.csv')
purchase_order_labels = read_csv_to_array('data/keywords/purchaseOrder.csv')
qst_amount_labels = read_csv_to_array('data/keywords/qstAmount.csv')
qst_number_labels = read_csv_to_array('data/keywords/qstNumber.csv')
reference_date_labels = read_csv_to_array('data/keywords/referenceDate.csv')
reference_number_labels = read_csv_to_array('data/keywords/referenceNumber.csv')
sub_total_amount_labels = read_csv_to_array('data/keywords/subTotalAmount.csv')
total_amount_labels = read_csv_to_array('data/keywords/totalAmount.csv')

def get_features():
    return ['match_qst_pattern','match_gst_pattern','amount', 'surrounded_by_gst_amount_label', 'surrounded_by_gst_number_label', 'surrounded_by_invoice_number_label', 'surrounded_by_purchase_order_label', 'surrounded_by_qst_amount_label', 'surrounded_by_qst_number_label', 'surrounded_by_reference_date_label', 'surrounded_by_reference_number_label', 'surrounded_by_sub_total_amount_label', 'surrounded_by_total_amount_label' ]

def pre_processing_image(img):
    return img

def text_recognition_from_image(img):
    return pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

#df['same_line_label_left'] = df.text.shift(periods=3).where(df.line_num.shift(periods=3) == df.line_num, '') + ' ' + df.text.shift(periods=2).where(df.line_num.shift(periods=2) == df.line_num, '') + ' ' + df.text.shift(periods=1).where(df.line_num.shift(periods=1) == df.line_num, '')
     
def surrounding_text_on_left(r, df: pd.DataFrame, accuracy):
    results = df.text.where((df.top >= (r.top - accuracy)) & (df.top <= (r.top + accuracy)) & ~pd.isnull(df.text), np.NAN)
    results = results.dropna()
    results = results.apply(lambda t: re.sub(r"\W+", '', t, flags=re.UNICODE))
    return results.str.cat(sep=' ').lower()

# Warning, when adding features, don't forget to add them to the training.py and app.py
def build_dataframe_from_ocr_data(ocr_data):
    df = pd.DataFrame(ocr_data)
    
    # Patterns
    df['match_qst_pattern'] = df.text.apply(lambda t: int(bool(re.search("\d{10}TQ\d{4}", t, re.IGNORECASE))))
    df['match_gst_pattern'] = df.text.apply(lambda t: int(bool(re.search("\d{9}", t, re.IGNORECASE))))
    df['amount'] = df.text.apply(lambda t: int(bool(re.search("^[+-]?[0-9]{1,3}(?:,?[0-9]{3})*[\.|,][0-9]{2} ?\$?$", t))))
    # Date pattern
    # 2024-02-01
    # 31/01/2024
     
    # Surrounding
    df['same_line_surrounding_text'] = df.apply(lambda r: surrounding_text_on_left(r, df, 10), axis=1)
    # Probably better way to do it

    # Contains
    df['surrounded_by_gst_amount_label'] = df['same_line_surrounding_text'].apply(lambda t: int(bool(any(token in t for token in gst_amount_labels))))
    df['surrounded_by_gst_number_label'] = df['same_line_surrounding_text'].apply(lambda t: int(bool(any(token in t for token in gst_number_labels))))
    df['surrounded_by_invoice_number_label'] = df['same_line_surrounding_text'].apply(lambda t: int(bool(any(token in t for token in invoice_number_labels))))
    df['surrounded_by_purchase_order_label'] = df['same_line_surrounding_text'].apply(lambda t: int(bool(any(token in t for token in purchase_order_labels))))
    df['surrounded_by_qst_amount_label'] = df['same_line_surrounding_text'].apply(lambda t: int(bool(any(token in t for token in qst_amount_labels))))
    df['surrounded_by_qst_number_label'] = df['same_line_surrounding_text'].apply(lambda t: int(bool(any(token in t for token in qst_number_labels))))
    df['surrounded_by_reference_date_label'] = df['same_line_surrounding_text'].apply(lambda t: int(bool(any(token in t for token in reference_date_labels))))
    df['surrounded_by_reference_number_label'] = df['same_line_surrounding_text'].apply(lambda t: int(bool(any(token in t for token in reference_number_labels))))
    df['surrounded_by_sub_total_amount_label'] = df['same_line_surrounding_text'].apply(lambda t: int(bool(any(token in t for token in sub_total_amount_labels))))
    df['surrounded_by_total_amount_label'] = df['same_line_surrounding_text'].apply(lambda t: int(bool(any(token in t for token in total_amount_labels))))
    
    return df