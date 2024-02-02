from flask import Flask, request
import pdf2image
import pickle
import cv2
import numpy as np
import uuid
import json
import pandas as pd

from processing import build_dataframe_from_ocr_data, pre_processing_image, text_recognition_from_image

DEBUG=True

model_filename = './src/invoice-model.pkl'
model = pickle.load(open(model_filename, 'rb'))

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

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
    image_page = pre_processing_image(image_page)

    # ----------------------------------------------------------------------------
    # STEP 3: Text recognition from OCR (using Tesseract)
    # ----------------------------------------------------------------------------
    ocr_data = text_recognition_from_image(image_page)
    
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
    df = build_dataframe_from_ocr_data(ocr_data)
    
    # Build content from OCR data
    ocr_content = ' '.join(df.text)
    
    result = {
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
    
    # Run the model for each row
    selected_features = ['match_qst_pattern','match_gst_pattern','amount']
    for i, row in df.iterrows():
        clean_row = row[selected_features]
        clean_row = clean_row.to_numpy().reshape(1, -1)
        prediction = model.predict(clean_row)

        match prediction:
            case 1: result['purchaseOrder'] = row.text
            case 2: result['gstNumber'] = row.text
            case 3: result['qstNumber'] = row.text
            case 4: result['totalAmount'] = row.text
    
    # ----------------------------------------------------------------------------
    # STEP 6: Return result
    # ----------------------------------------------------------------------------
    
    if DEBUG : df.to_csv('temp/' + request_id + '.csv', index=False)
    
    return app.response_class(
        response=json.dumps(result, ensure_ascii=False).encode('utf8'),
        status=200,
        mimetype='application/json'
    )
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8080', debug=DEBUG)