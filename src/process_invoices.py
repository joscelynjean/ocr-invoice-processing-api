#!/usr/bin/env python
"""PDF to text utility

This script will run the OCR on the extracted pdf image. This will save the data in CSV files.
"""

import glob
import os
from alive_progress import alive_bar
import pandas as pd
import pdf2image
import re
from processing import build_dataframe_from_ocr_data, pre_processing_image, text_recognition_from_image

DEBUG=False

training_dir_path = 'tests/training'

# Get all the PDF from the dataset directory
invoice_paths = glob.glob(training_dir_path + '/invoices/*.pdf')
if DEBUG : print(invoice_paths)

# For each PDF, extract the text
with alive_bar(len(invoice_paths)) as bar:
    for invoice_path in invoice_paths :
        
        # Get file name
        file_name = re.findall(r'[^\/]+(?=\.)',invoice_path)[0]
        
        # Convert PDF to images
        image_page_list = pdf2image.convert_from_path(invoice_path, fmt="jpeg")
        
        # Only supporting single page document
        image_page = image_page_list[0]
        
        # Clean the image
        image_page = pre_processing_image(image_page)
        
        # Text recognition with OCR
        ocr_data = text_recognition_from_image(image_page)
        
        # Build dataframe from the OCR data
        df = build_dataframe_from_ocr_data(ocr_data)
        
        # Save to CSV file, overwrite if exists
        csv_path = training_dir_path + '/datasets/' + file_name + '.csv'
        df.to_csv(csv_path, index=False, mode='w')
        
        # progress
        bar()
    
    