SPARK – Smart Product Analysis and Recommendation Kit
Project Overview

SPARK is a data-driven decision support system designed to analyze product performance, identify business opportunities, generate recommendations, and provide AI-powered explanations for decision-makers.

The project combines data processing, diagnostics, similarity analysis, bundle discovery, prescriptive analytics, and generative AI explanations within a unified Streamlit dashboard.

Features
Product performance diagnostics
Similarity analysis between products
Bundle recommendation generation
Prescriptive decision engine
AI-generated business explanations
Interactive Streamlit dashboard
End-to-end analytics pipeline
Dataset

The project uses the following datasets:

users.csv
products.csv
orders.csv
order_items.csv
reviews.csv
events.csv
Project Structure

SPARK_project/

├── app.py

├── config.py

├── run_pipeline.py

├── requirements.txt

├── src/

├── outputs/

├── data/

├── README.md

├── .env.example

└── .gitignore

Installation
Clone the repository
Create a virtual environment
Install dependencies

pip install -r requirements.txt

Create a .env file

GROQ_API_KEY=YOUR_API_KEY

Run Streamlit Application

streamlit run app.py

Run Full Pipeline

python run_pipeline.py

Technologies Used
Python
Streamlit
Pandas
NumPy
Scikit-learn
Plotly
MLxtend
Groq API
