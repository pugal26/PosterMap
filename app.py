from flask import Flask, request, jsonify
from flask_cors import CORS
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import pandas as pd
from fuzzywuzzy import process
import os
import re
import cv2
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins or specify your React app domain

# Set up pytesseract path (update the path if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Function to clean extracted text
def clean_text(text):
    text = re.sub(r'\W+', ' ', text).lower().strip()
    return text

# Function to preprocess the image for better OCR results
def preprocess_image(image_path):
    # Load the image with OpenCV
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoising
    img_denoised = cv2.fastNlMeansDenoising(img_gray, None, 30, 7, 21)

    # Apply Gaussian Blur to reduce noise
    img_blurred = cv2.GaussianBlur(img_denoised, (5, 5), 0)

    # Adaptive thresholding
    img_binary = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)

    # Convert back to PIL Image for pytesseract
    return Image.fromarray(img_binary)

# Function to process one image
def process_image(image_path, filenames):
    img = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(img, lang='eng', config=r'--oem 3 --psm 6')  # Adjust config
    cleaned_text = clean_text(extracted_text)
    match = process.extractOne(cleaned_text, filenames, scorer=process.fuzz.token_set_ratio)
    
    if match[1] < 70:
        partial_matches = process.extract(cleaned_text, filenames, scorer=process.fuzz.token_set_ratio)
        partial_matches = [(filename, score) for filename, score in partial_matches if score >= 50]
        return extracted_text, partial_matches if partial_matches else [("No match", 0)]
    
    return extracted_text, [match]


@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('images')
    
    # Load Excel file containing filenames and the new "file to rename" column
    excel_path = 'D:/Python/Poster Idendification/filename.xlsx'
    df = pd.read_excel(excel_path)

    # Extract the two columns: 'Filenames' and 'File to Rename'
    filenames = df.iloc[:, 0].tolist()  # Assuming filenames are in the first column
    file_to_rename = df.iloc[:, 1].tolist()  # Assuming 'File to Rename' is in the second column

    # Create a dictionary that maps filenames to their corresponding "file to rename"
    filename_mapping = {filenames[i]: file_to_rename[i] for i in range(len(filenames))}
    
    results = []
    
    for file in files:
        image_path = f'temp/{file.filename}'
        file.save(image_path)
        
        extracted_text, best_matches = process_image(image_path, filenames)
        
        if extracted_text:
            # Get the corresponding 'file to rename' for the matched filename
            matched_filenames = [match[0] for match in best_matches]
            rename_files = [filename_mapping.get(match, "Not available") for match in matched_filenames]
            
            # Debugging prints
            # print(f"Extracted Text: {extracted_text}")
            # print(f"Matched Filenames: {matched_filenames}")
            # print(f"File to Rename: {rename_files}")
            
            results.append({
                'Match Confidence': "; ".join([str(match[1]) for match in best_matches]),
                'Matched Filenames': "; ".join(matched_filenames),
                'File to Rename': "; ".join(rename_files),
                'Extracted Text': extracted_text,
                'Image Name': file.filename,
            })
        
        os.remove(image_path)  # Clean up the image file
    
    return jsonify(results)


if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')  # Create a temporary directory for image files
    app.run(port=5000)
