from flask import Flask, request
import pdf2image
import pytesseract

app = Flask(__name__)

def pdf_to_image(pdf_content):
    return pdf2image.convert_from_bytes(pdf_content, fmt="jpeg")

def image_to_text(image_content):
    return pytesseract.image_to_string(image_content)

@app.route("/invoice-processing-requests", methods=["POST"])
def process_invoice():
    
    # Read content of the file
    uploaded_file = request.files.get('invoice')
    file_content = uploaded_file.stream.read()
    
    # Convert PDF file to image
    image_content = pdf_to_image(file_content)
    
    # Extract text from image
    ocr_content = image_to_text(image_content[0])
    
    print(file_content)
    
    return {
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