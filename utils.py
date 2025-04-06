import os
from dotenv import load_dotenv
import google.generativeai as genai
from fpdf import FPDF

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_pdf(suggestions):
    """Generate a PDF from the suggestions text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Suggestions", ln=True, align='C')

    # Replace unsupported characters
    suggestions = suggestions.replace('\u2014', '-')  # Replace em dash with a regular dash
    suggestions = suggestions.replace('\u2013', '-')  # Replace en dash with a regular dash
    suggestions = suggestions.replace('\u2018', "'")  # Replace left single quote with a regular quote
    suggestions = suggestions.replace('\u2019', "'")  # Replace right single quote with a regular quote
    suggestions = suggestions.replace('\u201c', '"')  # Replace left double quote with a regular quote
    suggestions = suggestions.replace('\u201d', '"')  # Replace right double quote with a regular quote

    pdf.multi_cell(0, 10, txt=suggestions)  

    return pdf.output(dest='S').encode('latin1')

def call_gemini_api(data):
    """Calls Gemini API and returns suggestions based on provided mental health data."""
    prompt = f"""
Given the following mental health assessment data, generate a detailed yet empathetic natural language explanation of the predicted severity level. 

### Task:
1. Provide a **clear interpretation** of the results based on the given data.
2. Suggest **coping mechanisms** tailored to the individual's condition.
3. Recommend **potential next steps**, including professional consultation if necessary.

### Provided Information:
- **Age:** {data['age']}
- **Gender:** {data['gender']}
- **BMI:** {data['bmi']}
- **PHQ Score:** {data['phq_score']}
- **Depression Severity:** {data['anxiety_severity']}
- **Epworth Score (Daytime Sleepiness Level):** {data['epworth_score']}
- **GAD Score (Generalized Anxiety Disorder Level):** {data['gad_score']}
- **Predicted Severity:** {data['predicted_severity']}

### Expected Response:
- A **concise summary** of the mental health findings.
- **Personalized coping strategies** for managing symptoms.
- **Actionable next steps** for improving well-being, including professional support if needed.
"""

    response = model.generate_content(prompt)
    
    return response.text  
