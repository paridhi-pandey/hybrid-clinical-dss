import pytesseract
from PIL import Image
import spacy
import re
import os

# Configure tesseract for Windows if it exists in the default winget installation path
tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tess_path):
    pytesseract.pytesseract.tesseract_cmd = tess_path

# Try loading spaCy model gracefully
try:
    nlp = spacy.load("en_core_web_sm")
except (OSError, ImportError):
    nlp = None # Handle gracefully if spacy model is not downloaded

class PrescriptionAnalyzer:
    def __init__(self):
        pass

    def extract_text(self, image_path):
        """Extracts text from the given image using pytesseract."""
        try:
            image = Image.open(image_path)
            # Use basic config, preprocessing can be added for noisy data
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"Error extracting text: {e}"

    def analyze_text(self, text):
        """
        Simple NER using regex and heuristics to identify medicines and dosages.
        """
        structured_data = {
            "medicines": [],
            "dosages": [],
            "inferred_condition": "Unknown"
        }
        
        # Regex for common dosage patterns
        dosage_pattern = r'(\d+\s*(mg|ml|tablets?|capsules?|drops?|g|mcg|iu))'
        dosages = re.findall(dosage_pattern, text, flags=re.IGNORECASE)
        structured_data["dosages"] = list(set([d[0] for d in dosages]))
        
        words = text.split()
        
        # 1. Heuristic: Word before dosage
        for i, word in enumerate(words):
            if re.match(dosage_pattern, word, flags=re.IGNORECASE):
                if i > 0:
                    med_candidate = re.sub(r'[^a-zA-Z]', '', words[i-1])
                    if len(med_candidate) > 3 and med_candidate.lower() not in ["take", "one", "two", "tab", "cap"]:
                        structured_data["medicines"].append(med_candidate.capitalize())
        
        # 2. Heuristic: Common pharmacological suffixes & known meds
        med_suffixes = (
            "cillin", "mycin", "micin", "floxacin", "cycline", "azole",
            "pril", "olol", "statin", "sartan", "dipine", "mab", "nib",
            "vir", "tidine", "xaban", "asone", "olone", "pam", "lam",
            "fenac", "profen", "triptan", "gliptin", "gliflozin", "terol"
        )
        common_meds = {
            "paracetamol", "crocin", "dolo", "aspirin", "tylenol", 
            "advil", "metformin", "insulin", "lithium", "warfarin",
            "levothyroxine", "digoxin", "amiodarone", "clopidogrel",
            "cetirizine", "allegra", "loratadine"
        }
        
        for word in words:
            clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()
            if len(clean_word) > 3:
                if clean_word in common_meds or clean_word.endswith(med_suffixes):
                    structured_data["medicines"].append(clean_word.capitalize())
        
        # Remove duplicates
        structured_data["medicines"] = list(set(structured_data["medicines"]))
        
        # Basic condition inference (rule-based)
        text_lower = text.lower()
        if any(med in text_lower for med in ["paracetamol", "crocin", "dolo", "ibuprofen"]):
            structured_data["inferred_condition"] = "Fever / Pain"
        elif any(med in text_lower for med in ["amoxicillin", "azithromycin", "antibiotic"]):
            structured_data["inferred_condition"] = "Bacterial Infection"
        elif any(med in text_lower for med in ["cetirizine", "allegra", "loratadine"]):
             structured_data["inferred_condition"] = "Allergy / Cold"
             
        return structured_data
