import gradio as gr
from PyPDF2 import PdfReader
from io import BytesIO
import openai
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
import json
import plotly.express as px
from config import OPENAI_API_KEY
nltk.download('stopwords')
nltk.download('punkt')

openai.api_key = OPENAI_API_KEY

def echo_function(input_text):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.8,
        max_tokens=250,
        messages=[{"role": "system", "content": f"Given the following text of a CV, provide a brief summary of the candidate's professional background and list their top 5 skills along with a score out of 10 for each skill. I need this in JSON format For example: one key with the summary and other with the skills and score use those names. Insert CV Text Here -> {input_text}"}]
    )
    return completion.choices[0].message.content

def clean_text(text):
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    stop_words = set(stopwords.words('english'))
    words = [word for word in stripped if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

def process_and_summarize(file_path):
    # Read and extract text from the PDF
    text = extract_text(file_path)
    data = echo_function(clean_text(text))
    data = json.loads(data)

    # Extract skills and their scores
    skills_data = data.get("skills", {})
    skill_names = list(skills_data.keys())
    skill_scores = list(skills_data.values())

    # Create the plot
    fig = px.bar(x=skill_names, y=skill_scores, labels={'x': 'Skills', 'y': 'Scores'}, title='Candidate Skills and Scores')

    # Get the summary
    summary = data.get("summary", "No summary available")

    return fig, summary

# Gradio interface
interface = gr.Interface(
    fn=process_and_summarize,
    inputs=gr.File(label="Upload CV (PDF)", type="filepath"),
    outputs=[gr.Plot(label="Skills Bar Plot"), gr.Textbox(label="Professional Summary")]
)

interface.launch()

