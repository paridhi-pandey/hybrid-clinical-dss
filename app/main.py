import streamlit as st
import os
import sys
import joblib
import sys
import pkg_resources

st.write([pkg.key for pkg in pkg_resources.working_set])

# Add src to path safely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.disease_prediction import DiseasePredictor
from src.prescription_analysis import PrescriptionAnalyzer
from src.model_training import train_model

st.set_page_config(page_title="Hybrid Clinical DSS", page_icon="🏥", layout="wide")

# Ensure model exists or train it
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
if not os.path.exists(os.path.join(models_dir, 'pipeline_model.pkl')):
    st.info("First time setup: Training Machine Learning models from custom dataset... Please wait.")
    train_model()
    st.success("Model trained successfully!")

# Initialize singletons
@st.cache_resource
def get_predictor():
    return DiseasePredictor()

@st.cache_resource
def get_analyzer():
    return PrescriptionAnalyzer()

try:
    predictor = get_predictor()
    analyzer = get_analyzer()
    
    # Load feature metadata for dynamic UI
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
    numeric_features = joblib.load(os.path.join(models_dir, 'numeric_features.pkl'))
    categorical_features = joblib.load(os.path.join(models_dir, 'categorical_features.pkl'))
except Exception as e:
    st.error(f"Error initializing modules: {e}")
    st.stop()

st.title("🏥 Hybrid Clinical Decision Support System")

# Set up tabs
tab1, tab2 = st.tabs(["🩺 Disease Prediction", "📄 Prescription Analysis"])

# --- TAB 1: Disease Prediction ---
with tab1:
    st.header("🦠 Disease Prediction")
    st.write("Enter the required features to get a prediction based on our trained AI model.")
    
    pred_tab1, pred_tab2 = st.tabs(["📝 Text Description", "📋 Manual Feature Form"])
    
    with pred_tab1:
        st.write("### Describe your symptoms")
        symptom_text = st.text_area("Type your symptoms here (e.g. 'I have a severe fever, chills, and muscle ache')", height=150)
        
        if st.button("Predict from Text", key="predict_text_btn", type="primary"):
            if not symptom_text.strip():
                st.warning("Please enter some symptoms first.")
            else:
                user_inputs = {}
                text_lower = symptom_text.lower()
                
                # Match text to features
                matched_features = []
                for feature in feature_names:
                    if feature in numeric_features:
                        # Normalize feature name to match text (e.g. skin_rash -> skin rash)
                        feat_str = feature.lower().replace('_', ' ')
                        if feat_str in text_lower:
                            user_inputs[feature] = 1.0
                            matched_features.append(feat_str)
                        else:
                            user_inputs[feature] = 0.0
                    else:
                        user_inputs[feature] = ""
                        
                st.info(f"**Identified Symptoms:** {', '.join(matched_features) if matched_features else 'None detected'}")
                
                with st.spinner("Analyzing symptoms using trained Model..."):
                    predictions = predictor.predict(user_inputs)
                    
                    if predictions:
                        st.success("Prediction Complete!")
                        st.write("### Potential Diagnoses / Outcomes:")
                        for idx, pred in enumerate(predictions):
                            st.write(f"**{idx+1}. {pred['disease']}** (Confidence: {pred['confidence']}%)")
                            st.progress(pred['confidence'] / 100)
                    else:
                        st.warning("Could not find a confident prediction. Please try adding more specific symptoms.")

    with pred_tab2:
        st.write("### Manual Model Inputs")
        st.info("Input fields are dynamically generated based on the dataset features.")
        
        user_inputs = {}
        with st.expander("Fill Feature Values", expanded=True):
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            
            for i, feature in enumerate(feature_names):
                with cols[i % 3]:
                    if feature in numeric_features:
                        # Provide 0.0 as default for numerics
                        user_inputs[feature] = st.number_input(f"{feature}", value=0.0, format="%f", key=f"num_{feature}")
                    else:
                        # Provide empty string for categoricals
                        user_inputs[feature] = st.text_input(f"{feature}", value="", key=f"cat_{feature}")
        
        if st.button("Predict Outcome", key="predict_form_btn", type="primary"):
            with st.spinner("Analyzing inputs using trained Model..."):
                predictions = predictor.predict(user_inputs)
                
                if predictions:
                    st.success("Prediction Complete!")
                    st.write("### Potential Diagnoses / Outcomes:")
                    for idx, pred in enumerate(predictions):
                        st.write(f"**{idx+1}. {pred['disease']}** (Confidence: {pred['confidence']}%)")
                        st.progress(pred['confidence'] / 100)
                else:
                    st.warning("Could not find a confident prediction. Please adjust feature inputs.")

# --- TAB 2: Prescription Analysis ---
with tab2:
    st.header("📄 Prescription OCR & Analysis")
    st.write("Upload an image of a prescription. The system will extract text and identify medicines.")
    
    uploaded_file = st.file_uploader("Choose a prescription image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="Uploaded Prescription", use_container_width=True)
        
        with col2:
            if st.button("Analyze Prescription", key="ocr_btn", type="primary"):
                with st.spinner("Extracting text via OCR and applying NLP..."):
                    # Save temp file
                    temp_path = os.path.join(os.path.dirname(__file__), 'temp_img.png')
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract Text
                    raw_text = analyzer.extract_text(temp_path)
                    
                    # Remove temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                if raw_text and not raw_text.startswith("Error"):
                    st.subheader("📝 Extracted Raw Text")
                    st.text_area("OCR Output", raw_text, height=300)
                    
                    st.subheader("🔍 Structured Entities")
                    analysis = analyzer.analyze_text(raw_text)
                    
                    sub1, sub2 = st.columns(2)
                    with sub1:
                        st.write("**Identified Medicines:**")
                        if analysis["medicines"]:
                            for med in analysis["medicines"]:
                                st.write(f"- {med}")
                        else:
                            st.write("None found.")
                            
                    with sub2:
                        st.write("**Identified Dosages:**")
                        if analysis["dosages"]:
                            for dos in analysis["dosages"]:
                                st.write(f"- {dos}")
                        else:
                            st.write("None found.")
                            
                    st.info(f"**Inferred Condition Based on Medications:** {analysis['inferred_condition']}")
                    
                else:
                    st.error(f"Failed to extract text. Make sure Tesseract is installed. Details: {raw_text}")
