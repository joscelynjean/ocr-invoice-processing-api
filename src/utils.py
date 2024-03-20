from enum import Enum
import csv

class Classification(Enum):
    NONE = 0
    PURCHASE_ORDER = 1
    QST_NUMBER = 2
    GST_NUMBER = 3
    TOTAL_AMOUNT = 4
    SUB_TOTAL_AMOUNT = 5
    QST_AMOUNT = 6
    GST_AMOUNT = 7
    OTHER_TAX_AMOUNT = 8
    REFERENCE_DATE = 9
    REFERENCE_NUMBER = 10
    
    DOCUMENT_TYPE_OTHER = 100
    DOCUMENT_TYPE_INVOICE = 101
    DOCUMENT_TYPE_CREDIT_NOTE = 102
    
    
def read_csv_to_array(path):
    results = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader: # each row is a list
            results.append(row[0])
    return results
    