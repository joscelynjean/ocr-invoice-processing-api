#!/usr/bin/env python
"""Train and save model

This script will read all CSV from the dataset and build the model
"""

import glob
from alive_progress import alive_bar
import pandas as pd
import pickle
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from processing import get_features
from utils import Classification

DEBUG=True

dataset_dir_path = 'tests/training/datasets'

# Read expectation of the dataset
expectations = pd.read_csv('tests/training/expectation.csv')

# Get all CSV file pat hfrom the datasets directory
dataset_paths = glob.glob(dataset_dir_path + '/*[!.result].csv')

# Determine classification
def determine_classification(row, expectation) -> int:
    ocr_text = row.text.strip()
    if pd.isna(ocr_text): return Classification.NONE.value
    if expectation.purchase_order.size > 0 and ocr_text == expectation.purchase_order.values[0]: return Classification.PURCHASE_ORDER.value
    if expectation.qst_number.size > 0 and ocr_text == expectation.qst_number.values[0]: return Classification.QST_NUMBER.value
    if expectation.gst_number.size > 0 and ocr_text == expectation.gst_number.values[0]: return Classification.GST_NUMBER.value
    if expectation.total_amount.size > 0 and ocr_text == expectation.total_amount.values[0]: return Classification.TOTAL_AMOUNT.value
    if expectation.sub_total_amount.size > 0 and ocr_text == expectation.sub_total_amount.values[0]: return Classification.SUB_TOTAL_AMOUNT.value
    if expectation.gst_amount.size > 0 and ocr_text == expectation.gst_amount.values[0]: return Classification.GST_AMOUNT.value
    if expectation.qst_amount.size > 0 and ocr_text == expectation.qst_amount.values[0]: return Classification.QST_AMOUNT.value
    if expectation.other_tax_amount.size > 0 and ocr_text == expectation.other_tax_amount.values[0]: return Classification.OTHER_TAX_AMOUNT.value
    if expectation.reference_date.size > 0 and ocr_text == expectation.reference_date.values[0]: return Classification.REFERENCE_DATE.value
    if expectation.reference_number.size > 0 and ocr_text == expectation.reference_number.values[0]: return Classification.REFERENCE_NUMBER.value
    
    return Classification.NONE.value

# Build expected classification for each csv
df_data = []

# For each csv, calculate classification
with alive_bar(len(dataset_paths)) as bar:
    for csv_path in dataset_paths:
        # Extract filename, which will be used to find the corresponding expectation row
        file_name = re.findall(r'[^\/]+(?=\.)',csv_path)[0]
        # Read the CSV file
        df = pd.read_csv(csv_path)
        # Force text column as string
        df.text = df.text.astype(str)
        # Get the corresponding row
        expectation = expectations.loc[expectations.id == file_name]
        # Build classification column
        df['classification'] = df.apply(lambda row: determine_classification(row, expectation=expectation), axis=1)
        if DEBUG: df.to_csv(csv_path + '.result.csv', index=False, mode='w')
        # Add to our array
        df_data.append(df)
        # Update bar
        bar()

# Create a single dataframe from all the CSV files
df = pd.concat(df_data, ignore_index=True)

# Remove 90% of row having 0 as classification
df = df.drop(df[df.classification == 0].sample(frac=.99).index)

# Build training data
selected_features = get_features()
X = df[selected_features]
y = df.classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

# Train the model
model = LogisticRegression(multi_class='multinomial',solver='newton-cg', max_iter=10000)
model.fit(X_train.values, y_train)
predictions = model.predict(X_test.values)

print(classification_report(y_test, predictions))
# print('Predicted labels: ', predictions)
# print('Accuracy: ', accuracy_score(y_test, predictions))

# Pickle the model
model_filename = './src/invoice-model.pkl'
pickle.dump(model, open(model_filename, 'wb'))

