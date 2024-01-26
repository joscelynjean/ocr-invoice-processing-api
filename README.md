# ocr-invoice-processing-api
Experimenting a combinaison of Tesseract OCR, OpenCV and machine learning to extract field from an invoice.



# Process

1. Input file
2. Generate multiple images from source (original, without border, etc.)
3. Extract texts from images and original file
4. Use it as input for ML to get the right value

# Development

## Prerequisites

### MacOSX

1. Install tesseract using homebrew : `brew install tesseract`;