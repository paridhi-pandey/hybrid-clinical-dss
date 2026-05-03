# Hybrid Clinical Decision Support System Using NLP-Based Knowledge Extraction

## Step-by-step Implementation Plan

**Phase 1: Project Setup and Data Preparation**
1. Set up the project directory structure (`data/`, `models/`, `src/`, `app/`).
2. Create `requirements.txt` with necessary libraries.
3. Prepare a symptom-disease dataset (using a small, representative dataset).
4. Perform data preprocessing (cleaning, tokenization, feature extraction).

**Phase 2: Disease Prediction System**
1. Train a Random Forest classifier using `scikit-learn` on the symptom-disease dataset.
2. Develop a script (`src/model_training.py`) to train, evaluate, and save the model using `joblib`.
3. Create the inference pipeline (`src/disease_prediction.py`) to take user symptoms, apply the saved model, and output the predicted disease and confidence score.

**Phase 3: Medical Chatbot**
1. Develop basic intent classification and keyword mapping for typical queries (symptoms, precautions, advice).
2. Create the NLP logic (`src/chatbot.py`) using `nltk` or `spaCy`.
3. Integrate the Disease Prediction system so the chatbot can trigger predictions.

**Phase 4: Prescription Analysis Module (OCR + NER)**
1. Implement OCR using `pytesseract` to extract raw text from prescription images (`src/prescription_analysis.py`).
2. Apply NLP (`spaCy`) to extract structured data (Medicines) from the raw OCR text.
3. Infer the condition based on extracted medicines.

**Phase 5: User Interface Development (Streamlit)**
1. Develop `app/main.py` to serve as the unified Streamlit frontend.
2. Link the backend modules (prediction, chatbot, OCR) into the Streamlit app.
3. Run and test the Streamlit app.

---

## Folder Structure

```
project/
├── data/
│   └── dataset.csv                 # The dataset for ML
├── models/
│   └── rf_model.pkl                # Trained Random Forest Model
├── src/
│   ├── model_training.py           # Training logic
│   ├── disease_prediction.py       # Inference logic
│   ├── chatbot.py                  # Medical Chatbot logic
│   └── prescription_analysis.py    # OCR and NER logic
├── app/
│   └── main.py                     # Streamlit web application
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## Required Libraries

```
streamlit
scikit-learn
pandas
numpy
nltk
spacy
pytesseract
Pillow
joblib
```
*(Tesseract OCR needs to be installed on the system as well).*

---

## Sample Inputs and Outputs

**1. Disease Prediction System:**
*   **Input**: "I have a severe headache, fever, and nausea."
*   **Output**: Predicted Disease: Migraine (Confidence: 85%), Malaria (Confidence: 10%)

**2. Medical Chatbot:**
*   **Input**: "What are the precautions for COVID-19?"
*   **Output**: "To prevent COVID-19, wash your hands frequently, wear a mask, maintain social distancing, and get vaccinated."

**3. Prescription Analysis Module:**
*   **Input**: Image of a prescription containing: "Rx Paracetamol 500mg, 1 tablet twice a day."
*   **Output**: 
    *   Medicines: Paracetamol
    *   Dosage: 500mg
    *   Inferred Condition: Fever / Pain

---

## Suggestions for Improving Realism (Handling Noisy Medical Data)

1.  **Advanced NLP for Symptom Extraction**: Instead of count vectorizers on raw input, use advanced medical NER tools (e.g., BioBERT) to extract standardized medical concepts from free-text.
2.  **Addressing OCR Noise**: Prescriptions are hard to read. Enhance OCR accuracy using Image Preprocessing (binarization, denoising) via OpenCV.
3.  **Handling Uncertainty**: Apply probabilistic thresholds. If a prediction's confidence score is too low, the system should state: *"Symptoms are too ambiguous. Please consult a doctor."*
4.  **Multi-lingual Support**: Noisy data often contains vernacular terms for symptoms.
