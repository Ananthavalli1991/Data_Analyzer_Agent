# WARNING: This is a conceptual blueprint.
# 1. Do not hardcode API keys. Use environment variables for security.
# 2. You must have the required libraries installed:
#    pip install Flask google-generativeai Pillow
# 3. Replace the 'ocr_service' and its client with your actual implementation.

import os
import csv
import json
import re
import google.generativeai as genai
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# --- API Key Configuration (Use Environment Variables!) ---
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
OCR_API_KEY = os.environ.get("OCR_API_KEY")

# --- Initialize Gemini Model ---
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# --- File Handling and Tool Functions ---

def handle_text_file(file_content):
    """Reads and decodes a generic text file."""
    return file_content.decode('utf-8')

def handle_csv_file(file_content):
    """Reads a CSV file and formats it as a string for the LLM."""
    csv_string = ""
    csv_reader = csv.reader(BytesIO(file_content).read().decode('utf-8').splitlines())
    header = next(csv_reader)
    csv_string += ",".join(header) + "\n"
    for row in csv_reader:
        csv_string += ",".join(row) + "\n"
    return csv_string

def handle_json_file(file_content):
    """Reads a JSON file and formats it as a string for the LLM."""
    return json.dumps(json.loads(file_content), indent=2)

def handle_image_file(image_bytes):
    """
    Handles image files by sending them to a dedicated OCR API.
    You must replace this with your chosen OCR service's client code.
    """
    if not OCR_API_KEY:
        return "Error: OCR API key is not configured."

    url = "https://api.ocr.space/parse/image"
    payload = {
        "apikey": OCR_API_KEY,
        "language": "eng",
        "isOverlayRequired": False
    }
    files = {"file": ("image.jpg", image_bytes)}

    try:
        response = requests.post(url, files=files, data=payload, timeout=60)
        result = response.json()

        if result.get("IsErroredOnProcessing"):
            return f"OCR Error: {result.get('ErrorMessage', 'Unknown error')}"
        
        parsed_results = result.get("ParsedResults")
        if parsed_results:
            return parsed_results[0].get("ParsedText", "").strip()
        return "No text found in image."
    except Exception as e:
        return f"OCR API call failed: {str(e)}"


# A dictionary to route file extensions to their handler functions
FILE_HANDLERS = {
    '.txt': handle_text_file,
    '.csv': handle_csv_file,
    '.json': handle_json_file,
    '.png': handle_image_file,
    '.jpg': handle_image_file,
    '.jpeg': handle_image_file,
}

def analyze_with_gemini(questions, data_context):
    """
    Uses the Gemini API to analyze the full context and provide an answer.
    """
    full_prompt = (
        "You are a data analyst agent.\n"
        "Answer strictly as a JSON array.\n"
        "if answer is number, then return it as number.dont wrab number with quotes.\n"
        "Do not add explanations, text, or code fences.\n"
        "Your entire output must be a valid JSON array only.\n\n"
        "- Do NOT include any explanations, comments, or text before or after the JSON.\n"
        "- Do NOT use markdown formatting or code fences.\n"
        "- Stop immediately after the closing bracket `]`.\n\n"
        f"--- Questions to Answer ---\n{questions}\n\n"
        f"--- Data Context ---\n{data_context}"
    )

    print("Sending combined prompt to Gemini for analysis...")
    try:
        response = gemini_model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini API Error: {e}"

# --- Flask API Endpoint ---
app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def data_analyst_agent():
    # 1. Read 'questions.txt'
    if 'questions.txt' not in request.files:
        return jsonify({"error": "questions.txt file is required."}), 400
    
    questions = request.files['questions.txt'].read().decode('utf-8')
    data_context = ""

    # 2. Iterate through all other uploaded files
    for file_key, file_obj in request.files.items():
        if file_key == 'questions.txt':
            continue

        filename = file_obj.filename
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension in FILE_HANDLERS:
            print(f"Processing file: {filename}")
            file_content = file_obj.read()
            handler = FILE_HANDLERS[file_extension]
            processed_data = handler(file_content)
            data_context += f"\n--- Content from {filename} ---\n{processed_data}\n"
        else:
            print(f"Unsupported file type: {filename}. Skipping.")
    
    # 3. Pass all collected data to Gemini for analysis
    analysis_result = analyze_with_gemini(questions, data_context)
    return analysis_result, 200
        
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
