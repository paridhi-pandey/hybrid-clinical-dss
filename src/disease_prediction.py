import joblib
import os
import pandas as pd

class DiseasePredictor:
    def __init__(self):
        models_dir = os.path.join(os.path.dirname(__file__), '../models')
        self.pipeline = joblib.load(os.path.join(models_dir, 'pipeline_model.pkl'))
        self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        self.feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
        
    def predict(self, input_features_dict):
        """
        Predicts disease based on input feature dictionary.
        Returns predicted class and confidence scores.
        """
        # Ensure input data matches exactly the DataFrame structure expected by the pipeline
        input_df = pd.DataFrame([input_features_dict], columns=self.feature_names)
        
        # Predict probability
        probs = self.pipeline.predict_proba(input_df)[0]
        
        # Get the top predictions
        classes = self.pipeline.classes_
        
        results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        
        # Format the output using label encoder inverse transform
        predictions = []
        for encoded_class, prob in results[:3]:  # Top 3
            if prob > 0.05: # Only include if confidence is above 5%
                disease_name = self.label_encoder.inverse_transform([encoded_class])[0]
                predictions.append({"disease": disease_name, "confidence": round(prob * 100, 2)})
                
        return predictions
