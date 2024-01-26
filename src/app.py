from flask import Flask, request
import pdf2image
import pytesseract
import cv2
import numpy as np
import uuid

app = Flask(__name__)

def pdf_to_image(pdf_content):
    return pdf2image.convert_from_bytes(pdf_content, fmt="jpeg")

def image_to_text(image_content):
    return pytesseract.image_to_string(image_content)

@app.route("/invoice-processing-requests", methods=["POST"])
def process_invoice():
    request_id = str(uuid.uuid4())
    
    # Get input image
    
    # Read content of the file
    uploaded_file = request.files.get('invoice')
    file_content = uploaded_file.stream.read()
    
    # Convert PDF file to images. 1 image per pdf page
    image_page_list = pdf2image.convert_from_bytes(file_content, fmt="jpeg")
    
    # For now, only supporting single page document
    image_page = image_page_list[0]
        
    # Save extracted image to temp
    cv2.imwrite('temp/' + request_id + '.jpg', np.array(image_page))
    
    # Extract text from image
    ocr_content = pytesseract.image_to_string(image_page)
    f = open('temp/' + request_id + '.txt', 'a')
    f.write(ocr_content)
    f.close()
    
    # Extract data from image
    ocr_data = pytesseract.image_to_data(image_page, output_type=pytesseract.Output.DICT)
    
    # Rectangle for each word
    # @see https://www.docsumo.com/blog/tesseract-ocr
    annotated_image_page = np.array(image_page)
    for left, top, width, height, word_num in zip(ocr_data['left'], ocr_data['top'], ocr_data['width'], ocr_data['height'], ocr_data['word_num']):
        if word_num >= 1:
            annotated_image_page = cv2.rectangle(annotated_image_page, (left, top), (left + width, top + height), (0, 255, 0), 2)
    cv2.imwrite('temp/' + request_id + '_annotated.jpg', annotated_image_page)
    
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
    app.run(host='0.0.0.0', port='8080', debug=True)