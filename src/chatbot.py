import nltk
from nltk.tokenize import word_tokenize
import re

# Download required NLTK resources gracefully
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass # In case network fails

class MedicalChatbot:
    def __init__(self):
        # Basic intent keywords
        self.intents = {
            "greeting": ["hello", "hi", "hey", "greetings"],
            "precaution": ["precaution", "precautions", "prevent", "prevention", "avoid", "protect"],
            "advice": ["advice", "consult", "doctor", "help", "cure", "treatment"],
            "symptoms": ["symptom", "symptoms", "pain", "fever", "feel", "aching", "sick"]
        }
        
    def get_response(self, user_input):
        """
        Provides basic healthcare guidance based on intent.
        """
        user_input = user_input.lower()
        try:
            tokens = word_tokenize(user_input)
        except LookupError:
            # Fallback if punkt isn't available
            tokens = user_input.split()
        
        # Detect intent using regex word boundaries (more robust than exact token match)
        detected_intents = []
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                # \b checks for word boundaries, so "precaution" matches "precaution." or "precaution?"
                if re.search(r'\b' + re.escape(keyword) + r'\b', user_input):
                    detected_intents.append(intent)
                    break 
                
        # Generate response based on intent
        if "greeting" in detected_intents:
            return "Hello! I am your Medical Assistant. How can I help you today?"
            
        elif "precaution" in detected_intents:
            return "General precautions include washing hands regularly, staying hydrated, eating a balanced diet, and getting enough sleep. If you have specific symptoms, please let me know."
            
        elif "advice" in detected_intents:
            return "While I can provide basic information, I always recommend consulting a qualified healthcare professional or doctor for proper medical advice and diagnosis."
            
        elif "symptoms" in detected_intents:
            return "Please enter your symptoms in the Disease Prediction tab for a detailed analysis, or list them here and I can try to help."
            
        elif len(tokens) > 0:
            # Fallback for unrecognizable intents
            return "I'm not sure I understand. Could you rephrase? You can ask me for general precautions, medical advice, or tell me your symptoms."
            
        return "How can I help you?"
