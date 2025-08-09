import io
import json
import base64
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import List
from PIL import Image
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that uses LLMs to source, prepare, analyze, and visualize any data.",
)

# --- LLM and Data Processing Functions ---

async def call_gemini_api(prompt: str) -> dict:
    """
    Calls the Gemini API to get a structured analysis plan based on the prompt.
    This function uses a real API call to a Gemini model.
    """
    print(f"Prompt sent to Gemini API:\n{prompt}")

    # Note: The API key is an empty string here; the Canvas environment will inject it.
    apiKey =os.getenv("GEMINI_API_KEY")
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"
       
    
    # We add a generationConfig to force the model to respond with a JSON object.
    # The schema ensures the response matches the expected structure for our analysis engine.
    payload = {
        "contents": [
            {
                "role": "user", 
                "parts": [{ "text": f"Generate a JSON object for the following data analysis request:\n{prompt}\n\nStrictly adhere to this JSON schema for your response:\n\n{{\n  \"data_source\": \"<url_to_scrape_data>\",\n  \"tasks\": [\n    {{\n      \"type\": \"scrape_table\",\n      \"details\": {{\n        \"source_url\": \"<url_to_scrape_data>\",\n        \"table_description\": \"<a_natural_language_description_of_the_table_to_scrape>\"\n      }}\n    }},\n    {{\n      \"type\": \"question\",\n      \"details\": {{\n        \"question_text\": \"<the_question_to_answer>\",\n        \"calculation\": \"<a_single_line_valid_pandas_calculation_code_that_returns_a_value_e.g._'df['column'].mean()'_or_'df['column'].idxmax()'_>\"\n      }}\n    }},\n    {{\n      \"type\": \"visualization\",\n      \"details\": {{\n        \"question_text\": \"<the_question_to_answer>\",\n        \"chart_type\": \"<chart_type>\",\n        \"x_column\": \"<x_axis_column_name>\",\n        \"y_column\": \"<y_axis_column_name>\",\n        \"regression_line\": <true_or_false>,\n        \"regression_style\": \"<valid_matplotlib_line_style_string_e.g._'r--'_or_'b:'_>\",\n        \"image_format\": \"png\"\n      }}\n    }}\n  ]\n}}" }]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
        }
    }

    try:
        response = requests.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        result = response.json()
        
        if result and "candidates" in result and len(result["candidates"]) > 0:
            json_text = result["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(json_text)
        else:
            raise ValueError("Invalid response from Gemini API")
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get a plan from Gemini API: {e}")


def find_table_by_description(soup, description):
    """
    Finds the best-matching table in the HTML soup based on a natural language description.
    This is a placeholder and would require a more sophisticated LLM-based approach
    for robust matching. For now, we'll just return the first table.
    """
    return soup.find("table")


def analyze_and_visualize_data(analysis_plan: dict, files: List[bytes]) -> list:
    """
    This is the core analysis and visualization engine.
    It now interprets a generic analysis plan to perform the specified tasks.
    It uses pandas and matplotlib to handle data manipulation and plotting.
    """
    df = pd.DataFrame()
    answers = []
    
    try:
        # Loop through each task in the analysis plan
        for task in analysis_plan.get("tasks", []):
            task_type = task.get("type")
            details = task.get("details", {})

            if task_type == "scrape_table":
                # Scrape data from a URL using BeautifulSoup
                url = details.get("source_url")
                if not url:
                    continue

                print(f"Scraping data from: {url}")
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Use a placeholder to find the relevant table
                # A more advanced version would use the 'table_description'
                # from the analysis plan to find a specific table.
                html_table = soup.find('table')
                
                if not html_table:
                    raise ValueError("No tables found on the page.")
                
                # Fixed: Wrap the HTML string in StringIO to suppress the FutureWarning
                df = pd.read_html(io.StringIO(str(html_table)))[0]

                # Handle potential multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                
                # Clean up column names by removing special characters and extra spaces
                df.columns = [col.replace(u'\xa0', u' ').split('[')[0].strip() for col in df.columns]

                # Print the cleaned column names for debugging
                print("Cleaned DataFrame Columns:", df.columns)

                # Perform targeted cleaning and conversion for specific columns
                numeric_cols = ['Worldwide gross', 'Year', 'Rank', 'Peak']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(
                            df[col].astype(str).str.replace(r'[$,%A-Za-z]', '', regex=True),
                            errors='coerce'
                        )
                
                # Only drop rows if the columns we're using actually exist.
                existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
                if existing_numeric_cols:
                    df.dropna(subset=existing_numeric_cols, inplace=True)

                if df.empty:
                    answers.append("Error: The DataFrame is empty after cleaning. Cannot perform calculations.")
                    continue
                
                print("Cleaned DataFrame Head:", df.head())
            
            elif task_type == "question":
                if df.empty:
                    answers.append("Error: Cannot perform a question task on an empty DataFrame.")
                    continue
                calculation_string = details.get("calculation")
                if not calculation_string:
                    continue

                local_vars = {'df': df, 'pd': pd, 'np': np}
                try:
                    # Added a check to make sure the columns in the calculation string exist in the DataFrame
                    required_columns = [col for col in ['Worldwide gross', 'Year'] if col in calculation_string]
                    if not all(col in df.columns for col in required_columns):
                        answers.append(f"Error: Missing columns in DataFrame for calculation: {required_columns}")
                        continue
                    
                    result = eval(calculation_string, {"__builtins__": None}, local_vars)
                    answers.append(result)
                except KeyError as key_e:
                    print(f"KeyError during eval: {key_e}. Attempting to fix the column name.")
                    if "'Film'" in str(key_e) and 'Title' in df.columns:
                        fixed_calculation_string = calculation_string.replace("'Film'", "'Title'")
                        try:
                            result = eval(fixed_calculation_string, {"__builtins__": None}, local_vars)
                            answers.append(result)
                        except Exception as fix_e:
                            answers.append(f"Error evaluating corrected question: {fix_e}")
                    else:
                        answers.append(f"Error evaluating question: {key_e}")
                except Exception as eval_e:
                    print(f"Error during eval of '{calculation_string}': {eval_e}")
                    answers.append(f"Error evaluating question: {eval_e}")

            elif task_type == "visualization":
                if df.empty:
                    answers.append("Error: Cannot perform a visualization task on an empty DataFrame.")
                    continue
                
                x_col = details.get("x_column")
                y_col = details.get("y_column")
                
                if not (x_col and y_col and x_col in df.columns and y_col in df.columns):
                    answers.append("Error: Missing or invalid x_column or y_column for visualization.")
                    continue

                fig, ax = plt.subplots()
                ax.scatter(df[x_col], df[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'{x_col} vs. {y_col}')

                if details.get("regression_line"):
                    z = np.polyfit(df[x_col], df[y_col], 1)
                    p = np.poly1d(z)
                    ax.plot(df[x_col], p(df[x_col]), details.get("regression_style", "r--"))

                buf = io.BytesIO()
                fig.savefig(buf, format=details.get("image_format", "png"), bbox_inches='tight')
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                img_uri = f"data:image/{details.get('image_format', 'png')};base64,{img_b64}"
                
                if len(img_b64) > 100000:
                    print(f"Warning: Image size exceeds 100,000 bytes. Actual size: {len(img_b64)} bytes.")

                plt.close(fig)
                answers.append(img_uri)
        
        # Convert NumPy types to standard Python types for JSON serialization.
        for i, answer in enumerate(answers):
            if isinstance(answer, (np.int64, np.int32, np.float64, np.float32)):
                answers[i] = answer.item()
            elif isinstance(answer, pd.Series):
                answers[i] = answer.to_list()
        
        return answers
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        return [f"An error occurred during analysis: {str(e)}"]

# The API endpoint
@app.post("/api/")
async def data_analyst_agent(questions_file: UploadFile = File(...), other_files: List[UploadFile] = File([])):
    """
    API endpoint for the Data Analyst Agent.
    Accepts a POST request with a 'questions.txt' file and optional attachments.
    """
    try:
        questions_content = await questions_file.read()
        prompt_content = questions_content.decode('utf-8')

        analysis_plan = await call_gemini_api(prompt_content)
        
        analysis_result = analyze_and_visualize_data(analysis_plan, [])
        
        return JSONResponse(content=analysis_result)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
