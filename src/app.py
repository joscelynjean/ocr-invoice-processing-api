from flask import Flask, request
import pdf2image
import pytesseract
import cv2
import numpy as np
import uuid
import pandas as pd
import re

DEBUG=True

app = Flask(__name__)

def pdf_to_image(pdf_content):
    return pdf2image.convert_from_bytes(pdf_content, fmt="jpeg")

def image_to_text(image_content):
    return pytesseract.image_to_string(image_content)

@app.route("/invoice-processing-requests", methods=["POST"])
def process_invoice():
    
    # Define request ID
    request_id = str(uuid.uuid4())
    
    # ----------------------------------------------------------------------------
    # STEP 1 : Get the corresponding images from the PDF that has been sent
    # ----------------------------------------------------------------------------
    
    # Read content of the file in the request
    uploaded_file = request.files.get('invoice')
    file_content = uploaded_file.stream.read()
    
    # Convert PDF file to images. 1 image per pdf page
    image_page_list = pdf2image.convert_from_bytes(file_content, fmt="jpeg")
    
    # For now, only supporting single page document
    image_page = image_page_list[0]
    
    # Save extracted image to temp
    if DEBUG : cv2.imwrite('temp/' + request_id + '.jpg', np.array(image_page))
    
    # ----------------------------------------------------------------------------
    # STEP 2 : Pre-processing of the image
    # ----------------------------------------------------------------------------
    
    # Here, should use technique to make the image better for the OCR (ex: remove border, contrast)
    # TODO : Improving image for recognition

    # ----------------------------------------------------------------------------
    # STEP 3: Text recognition from OCR (using Tesseract)
    # ----------------------------------------------------------------------------
    
    # TODO : Remove content and build it from data instead
    ocr_content = pytesseract.image_to_string(image_page)
    ocr_data = pytesseract.image_to_data(image_page, output_type=pytesseract.Output.DICT)
    
    # ----------------------------------------------------------------------------
    # STEP 4: Post-processing
    # ----------------------------------------------------------------------------
    
    # Draw Rectangle for each word, for debugging purpose
    # @see https://www.docsumo.com/blog/tesseract-ocr
    if DEBUG :
        annotated_image_page = np.array(image_page)
        for left, top, width, height, word_num in zip(ocr_data['left'], ocr_data['top'], ocr_data['width'], ocr_data['height'], ocr_data['word_num']):
            if word_num >= 1:
                annotated_image_page = cv2.rectangle(annotated_image_page, (left, top), (left + width, top + height), (0, 255, 0), 2)
        cv2.imwrite('temp/' + request_id + '_annotated.jpg', annotated_image_page)
        
    # ----------------------------------------------------------------------------
    # STEP 5: Classification by using Machine learning
    # ----------------------------------------------------------------------------
    
    df = pd.DataFrame(ocr_data)
    
    df['match_qst_pattern'] = df.text.apply(lambda t: int(bool(re.search("\d{10}TQ\d{4}", t, re.IGNORECASE))))
    df['match_gst_pattern'] = df.text.apply(lambda t: int(bool(re.search("\d{9}", t, re.IGNORECASE))))
    df['amount'] = df.text.apply(lambda t: int(bool(re.search("^[+-]?[0-9]{1,3}(?:,?[0-9]{3})*[\.|,][0-9]{2} ?\$?$", t))))
    
    # ----------------------------------------------------------------------------
    # STEP 6: Return result
    # ----------------------------------------------------------------------------
    
    if DEBUG :
        df.to_csv('temp/' + request_id + '.csv', index=True)
        with open('temp/' + request_id + '.txt', 'w') as f:
            for level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text in zip(ocr_data['level'], ocr_data['page_num'], ocr_data['block_num'], ocr_data['par_num'], ocr_data['line_num'], ocr_data['word_num'], ocr_data['left'], ocr_data['top'], ocr_data['width'], ocr_data['height'], ocr_data['conf'], ocr_data['text']):
                f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'.format(level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text))
    
    return {
        'id': request_id,
        'documentType': '',
        'gstAmount': '',
        'qstAmount': '',
        'taxAmount': '',
        'totalAmount': '',
        'purchaseOrder': '',
        'gstNumber': '',
        'qstNumber': '',
        'ocrResult': ocr_content
    }
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8080', debug=DEBUG)