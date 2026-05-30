# Retinal AI — Diabetic Retinopathy Grading System

An AI-powered web application for grading diabetic retinopathy severity 
from retinal fundus photographs using MobileNetV2 transfer learning.

## About
This tool classifies retinal images into 5 DR severity grades:
- Grade 0: No DR
- Grade 1: Mild
- Grade 2: Moderate
- Grade 3: Severe
- Grade 4: Proliferative DR

## Model Performance
| Model | Accuracy | Kappa Score |
|---|---|---|
| Basic CNN (baseline) | 39% | 0.29 |
| MobileNetV2 (final) | 68% | 0.71 |

## Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- Plotly

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## Disclaimer
This tool is intended for research and educational purposes only.
It is not a substitute for clinical diagnosis by a qualified ophthalmologist.