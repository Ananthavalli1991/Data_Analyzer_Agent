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

