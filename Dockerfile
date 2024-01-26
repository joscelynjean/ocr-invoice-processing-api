FROM python:3.11-alpine

WORKDIR /app
COPY . /app

RUN apk add poppler-utils tesseract-ocr tesseract-ocr-data-eng tesseract-ocr-data-fra

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "./src/app.py"]