<img width="1519" height="788" alt="image" src="https://github.com/user-attachments/assets/cb096ed0-6de6-4e63-b206-3e49080568c0" />
<img width="1421" height="757" alt="Screenshot 2026-06-07 205727" src="https://github.com/user-attachments/assets/00d370d5-1bac-4ecc-8bf7-65d65133e277" />
<img width="1433" height="808" alt="Screenshot 2026-06-07 205205" src="https://github.com/user-attachments/assets/1a0ff6d5-a393-4b82-afe8-a84415f3f62c" />


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
