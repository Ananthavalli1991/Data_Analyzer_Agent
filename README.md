# Data_Analyzer_Agent
This repository was created for sourcing , processing and analyzing any data 

An API that uses Large Language Models (LLMs) to source, prepare, analyze, and visualize any tabular data from web scraping or uploaded files.  
Built with FastAPI, it accepts natural language data analysis requests and returns answers and visualizations.

---

## Features

- Accepts a `questions.txt` file describing data analysis tasks.
- Uses LLM (Gemini API / Google Gen AI) to generate a structured analysis plan.
- Scrapes tables from websites using BeautifulSoup.
- Performs data analysis with Pandas and NumPy.
- Generates charts and visualizations with Matplotlib.
- Returns answers and base64-encoded images as API JSON response.

---

## Tech Stack

- Python 3.10+
- FastAPI
- Uvicorn
- Requests & BeautifulSoup4
- Pandas & NumPy
- Matplotlib
- Google Gemini / GenAI API (via `genai` SDK or HTTP calls)
- python-dotenv for local environment variables

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Gemini API key from Google Cloud (set as `GEMINI_API_KEY`)

### Installation

1. Clone the repo:

```bash
git clone https://github.com/Ananthavalli1991/Data_Analyzer_Agent.git
cd Data_Analyzer_Agent

2. Create a Virtual Environment

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

On macOS or Linux:

source venv/bin/activate
3. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt
4. Configure Environment Variables

Create a .env file in the root directory of the project.

Add the API keys required by your configuration:

GEMINI_API_KEY=your_gemini_api_key
OCR_API_KEY=your_ocr_api_key

Replace the example values with your actual API keys. If OCR is not configured or required, omit OCR_API_KEY as appropriate for your implementation.

Security note: Never commit your .env file or API keys to GitHub. Keep credentials private and use environment variables in your deployment environment.

Usage
1. Start the Application

Run the Flask application:

python app.py

If the application starts successfully using Flask's development server, it will typically be available at:

http://localhost:5000
2. Prepare the Input Files

Create a questions.txt file containing the analytical instructions you want the application to perform.

You can also provide supported data files, such as CSV, JSON, text, or image files, depending on the task.

Example questions.txt
Scrape the list of highest-grossing films from Wikipedia.

Source:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and return a JSON array of strings:

1. How many $2 billion movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 billion?
3. What is the correlation between Rank and Peak?
4. Draw a scatterplot of Rank and Peak with a dotted red regression line.
   Return the chart as a base64-encoded PNG data URI in the format:
   data:image/png;base64,<encoded-image>

Keep the encoded image under 100,000 bytes.

This example demonstrates the intended use of natural-language instructions. The accuracy of the answers depends on the data collected, the analysis performed, and the application's implementation.

3. Send a Request to the API

The API endpoint is:

POST /api/
Send Only questions.txt

Run the following command from the directory containing your input file:

curl -X POST "http://localhost:5000/api/" \
  -F "questions.txt=@questions.txt"
Send Multiple Files

To include additional input files, use:

curl -X POST "http://localhost:5000/api/" \
  -F "questions.txt=@questions.txt" \
  -F "image.png=@image.png" \
  -F "data.csv=@data.csv"

Replace the example filenames with the files you want to analyze.

4. API Response

The API is designed to return a JSON array containing answers to the submitted questions.

An illustrative response might look like this:

[
  "Example analytical answer",
  "Example numerical result",
  "data:image/png;base64,<encoded-image>"
]

The actual response depends on the questions, input data, and analysis performed.

Deployment

The repository includes a render.yaml configuration for deployment on Render.

Deployment requires the appropriate dependencies, environment variables, and application start command.

Before deploying, verify that:

All required dependencies are listed in requirements.txt.
The application uses the correct start command.
API keys are configured as environment variables in Render.
No secrets are committed to the repository.
The deployed API accepts the expected multipart form-data request.

For more information, see the deployment configuration in this repository.

Limitations
Analysis quality depends on the input data and the instructions provided.
AI-generated answers may contain errors and should be validated.
Web scraping depends on website accessibility and page structure.
Image text extraction may require a valid external OCR API key.
API usage may be subject to provider quotas, latency, and costs.
Supported input formats and analytical capabilities depend on the current implementation.
Future Improvements
Add automated tests for API endpoints and data-processing workflows.
Improve validation of uploaded files and analytical instructions.
Strengthen error handling for invalid inputs and external API failures.
Add more sample requests and verified response examples.
Improve documentation for configuration and deployment.
Expand supported analytical workflows where appropriate.
License

This project is licensed under the MIT License. See the LICENSE file for details.

Author

Ananthavalli
