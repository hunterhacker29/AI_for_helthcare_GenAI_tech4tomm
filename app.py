
# import streamlit as st
# import pickle
# import re
# import requests
# import json
# import io
# import concurrent.futures
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Define raw file URLs for the models
# model_url = "https://raw.githubusercontent.com/hunterhacker29/AI_for_helthcare_GenAI_tech4tomm/main/logistic_regression_model.pkl"
# tfidf_url = "https://raw.githubusercontent.com/hunterhacker29/AI_for_helthcare_GenAI_tech4tomm/main/tfidf_vectorizer.pkl"

# def load_models():
#     # Fetch the model files directly from the URLs
#     model_response = requests.get(model_url)
#     tfidf_response = requests.get(tfidf_url)

#     if model_response.status_code == 200 and tfidf_response.status_code == 200:
#         # Load models directly from the response content
#         loaded_model = pickle.load(io.BytesIO(model_response.content))
#         loaded_tfidf = pickle.load(io.BytesIO(tfidf_response.content))
#         return loaded_model, loaded_tfidf
#     else:
#         raise Exception("Error fetching models from GitHub")

# # Load the pre-trained models
# try:
#     loaded_model, loaded_tfidf = load_models()
# except Exception as e:
#     st.error(f"Failed to load models: {e}")
#     # You can handle errors or provide fallback logic if necessary

# def preprocess_input_text(input_text):
#     processed_text = []
#     for text in input_text:
#         text = text.lower()  # Convert text to lowercase
#         text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#         text = re.sub(r'\d+', '', text)  # Remove digits
#         text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
#         processed_text.append(text)
#     return processed_text

# def get_model_response(user_input):
#     processed_text = preprocess_input_text([user_input])
    
#     # Ensure the vectorizer is transformed on the correctly preprocessed text
#     new_text_tfidf = loaded_tfidf.transform(processed_text)
    
#     predicted_label = loaded_model.predict(new_text_tfidf)
#     return predicted_label[0]

# def get_gemini_response(predicted_label):
#     API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
#     API_KEY = "AIzaSyAWfOaA_zYrilK7uQeuC3Mh0552PaoMRzo"  # Replace with your actual API key

#     headers = {
#         'Content-Type': 'application/json',
#     }

#     user_input = f"Predicted label: {predicted_label}. Give some home remedies and consulting to cure this disease"

#     data = {
#         "contents": [{
#             "parts": [{"text": user_input}]
#         }]
#     }

#     try:
#         # Send the request to the Gemini API
#         response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, data=json.dumps(data))
#         response.raise_for_status()  # Raise an error if the request fails
#         api_response = response.json()  # Parse the JSON response

#         if 'candidates' in api_response and len(api_response['candidates']) > 0:
#             return api_response['candidates'][0]['content']['parts'][0]['text']
#         else:
#             return "I'm sorry, I didn't get a valid response."
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error contacting Gemini API: {e}")
#         return "I'm sorry, there was an error processing your request."

# # Function to check if the input contains symptom-related terms
# def contains_symptom_keywords(query):
#     symptom_keywords = [
#         "thirst", "urination", "fatigue", "weight loss", "vision", "wounds", "healing", "blurred vision", "slow-healing",
#         "headache", "fever", "joint pain", "rash", "nausea", "vomiting", "muscle pain", "cough", "chest pain", 
#         "abdominal pain", "shortness of breath", "swelling", "dizziness", "loss of appetite", "diarrhea", "jaundice",
#         "dark urine", "skin lesions", "numbness", "pain during intercourse", "pelvic pain", "leg cramps", "hydrophobia",
#         "swollen lymph nodes", "wheezing", "difficulty swallowing", "hoarseness", "sore throat", "stiff neck", 
#         "body aches", "irregular heartbeat", "diabetic", "immune system", "contagious", "muscle weakness", "weight gain",
#         "blood in sputum", "persistent cough", "loss of sensation", "blurry vision", "irritability", "delayed growth","ache","aches","arm", "leg", "pain"
#     ]
    
#     query = query.lower()
#     return any(keyword in query for keyword in symptom_keywords)

# st.title("Ask SymptomScout")

# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# for messages in st.session_state.messages:
#     st.chat_message(messages['role']).markdown(messages['content'])

# # User input
# prompt = st.chat_input("Ask SymptomScout")

# if prompt:
#     st.chat_message("user").markdown(prompt)
#     st.session_state.messages.append({'role': 'user', 'content': prompt})

#     if contains_symptom_keywords(prompt):  # Check if the input contains symptoms
#         with st.spinner('Processing your request... Please wait while we analyze the symptoms.'):

#             # Use ThreadPoolExecutor to run both tasks in parallel
#             with concurrent.futures.ThreadPoolExecutor() as executor:
#                 # Run the model prediction and Gemini API call in parallel
#                 future_label = executor.submit(get_model_response, prompt)
#                 predicted_label = future_label.result()  # This will block until the prediction is done
                
#                 # Feed the predicted label to the API for information, remedies, and recommendations
#                 future_gemini_response = executor.submit(get_gemini_response, predicted_label)
#                 gemini_response = future_gemini_response.result()  # This will block until the response is ready

#         # Append doctor and hospital details to the response shown to the user
#         doctor_info = f"\n\nDoctor Specialist in {predicted_label} Treatment:\n\nDr. Ravi Kumar\nSpecialization: Infectious Disease Specialist, {predicted_label} Treatment\nExperience: 10+ years in treating mosquito-borne diseases, specializing in dengue fever management.\nConsultation Fee: ₹1000 (Approx.)\nLocation: Fortis Healthcare, Marine Drive, Mumbai, Maharashtra, India.\nPhone: +91 98234 56789\n\nNearby Hospitals:\n\nBreach Candy Hospital\nAddress: Breach Candy, Mumbai, Maharashtra, India\nContact: +91 22 1234 5678\n\nJaslok Hospital\nAddress: 15, Dr. Deshmukh Marg, Mumbai, Maharashtra, India\nContact: +91 22 6660 1234\n\nKokilaben Dhirubhai Ambani Hospital\nAddress: Andheri West, Mumbai, Maharashtra, India\nContact: +91 22 4260 6000"

#         # Combine the Gemini response with the doctor details
#         final_response = gemini_response + doctor_info

#         st.chat_message("SymptomScout").markdown(f"Predicted label: The given symptoms is of {predicted_label}")
#         st.session_state.messages.append({'role': 'SymptomScout', 'content': f"Predicted label: {predicted_label}"})

#         st.chat_message("SymptomScout").markdown(final_response)
#         st.session_state.messages.append({'role': 'SymptomScout', 'content': final_response})
#     else:
#         st.chat_message("SymptomScout").markdown("Please enter your symptoms so I can assist you better. I am a medical-focused AI here to help!")
#         st.session_state.messages.append({'role': 'SymptomScout', 'content': "Please enter your symptoms so I can assist you better. I am a medical-focused AI here to help!"})



import streamlit as st
import os
import requests
import json
import pickle
import re
import random
import concurrent.futures
import numpy as np
import io

# Function to initialize session state variables
def initialize_session_state():
    if 'global_array' not in st.session_state:
        st.session_state.global_array = [0] * 40
    if 'predicted_label' not in st.session_state:
        st.session_state.predicted_label = None
    if 'update_counter' not in st.session_state:
        st.session_state.update_counter = 0
    if 'messages' not in st.session_state:
        st.session_state.messages = []

# Initialize session state variables
initialize_session_state()

symptoms = [
    "hot flashes", "pelvic pain", "problems with movement", "retention of urine", "weakness", "lower body pain",
    "heartburn", "nausea", "joint pain", "arm pain", "disturbance of memory", "vomiting", "back pain", "low back pain",
    "abusing alcohol", "chills", "headache", "shortness of breath", "dizziness", "leg pain", "diarrhea", "skin irritation",
    "sharp abdominal pain", "sore throat", "ache all over", "sharp chest pain", "lower abdominal pain", "skin rash",
    "decreased appetite", "cough", "side pain", "neck pain", "burning abdominal pain", "fever", "difficulty breathing",
    "allergic reaction", "coryza", "chest tightness", "nasal congestion", "skin swelling"
]

symptom_keywords = [
    'skin problems', 'side pain', 'irritability', 'lower body pain', 'slow-healing', 'wounds', 'weight loss', 'blood in sputum', 'weakness', 'rapid heartbeat', 'hoarseness', 'neck pain', 'depression', 'skin rash', 'anxiety', 'skin irritation', 'high blood pressure', 'wheezing', 'retention of urine', 'immune system', 'coryza', 'cough', 'decreased appetite', 'shortness of breath', 'chest pain', 'blurred vision', 'numbness', 'irregular heartbeat', 'personality changes', 'abusing alcohol', 'diabetic', 'low libido', 'chest tightness', 'muscle weakness', 'erectile dysfunction', 'diarrhea', 'fatigue', 'muscle pain', 'lower abdominal pain', 'blurry vision', 'dizziness', 'abdominal pain', 'swollen lymph nodes', 'appetite changes', 'jaundice', 'swelling', 'poor coordination', 'problems with movement', 'insomnia', 'restlessness', 'painful periods', 'muscle tension', 'low back pain', 'burning abdominal pain', 'urinary problems', 'nasal congestion', 'nausea', 'leg cramps', 'hot flashes', 'ache all over', 'contagious', 'sweating', 'delayed growth', 'seizures', 'pregnancy', 'back pain', 'pelvic pain', 'body aches', 'memory loss', 'headaches', 'infertility', 'vomiting', 'skin lesions', 'weight gain', 'vaginal dryness', 'fainting', 'leg pain', 'allergic reaction', 'fever', 'thirst', 'sore throat', 'confusion', 'arm pain', 'skin swelling', 'menopause', 'heartburn', 'persistent cough', 'difficulty breathing', 'dark urine', 'joint pain', 'sharp chest pain', 'urination', 'stiff neck', 'hallucinations', 'mood swings', 'pain', 'difficulty swallowing', 'chills', 'rash', 'sharp abdominal pain', 'digestive problems', 'sexual problems', 'night sweats', 'headache', 'vision', 'hello', 'hair loss', 'difficulty concentrating', 'tremors', 'pain during intercourse', 'loss of appetite', 'tingling', 'hydrophobia', 'loss of sensation', 'disturbance of memory', 'healing', 'ache', 'arm', 'leg', 'aches', "weak", "hi", "chest", "tight", "feels", "dizzy"
]

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
API_KEY = "AIzaSyAWfOaA_zYrilK7uQeuC3Mh0552PaoMRzo"

st.set_page_config(
    page_title="SymptomScout",
    page_icon="🩺",
    layout="centered"
)

# Function to load models from URLs
def load_models():
    # Define raw file URLs for the models
    nlp_model_url = "https://github.com/Radical-Ghost/SymptomScout/blob/main/src/models/NLPModel.pkl?raw=true"
    vectorizer_url = "https://github.com/Radical-Ghost/SymptomScout/blob/main/src/models/Vectorizer.pkl?raw=true"
    classifier_model_url = "https://github.com/Radical-Ghost/SymptomScout/blob/main/src/models/DiseaseClassifier.pkl?raw=true"
    
    # Fetch the model files directly from the URLs
    nlp_model_response = requests.get(nlp_model_url)
    vectorizer_response = requests.get(vectorizer_url)
    classifier_model_response = requests.get(classifier_model_url)

    if nlp_model_response.status_code == 200 and vectorizer_response.status_code == 200 and classifier_model_response.status_code == 200:
        # Load models directly from the response content
        nlp_model = pickle.load(io.BytesIO(nlp_model_response.content))
        vectorizer = pickle.load(io.BytesIO(vectorizer_response.content))
        classifier_model = pickle.load(io.BytesIO(classifier_model_response.content))
        return nlp_model, vectorizer, classifier_model
    else:
        raise Exception("Error fetching models from GitHub")

# Load the pre-trained models
try:
    nlp_model, vectorizer, classifier_model = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
# Preprocess input text
def preprocess_input_text(input_text):
    processed_text = []
    for text in input_text:
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        processed_text.append(text)
    return processed_text

# Check if the input contains symptom-related terms
def contains_symptom_keywords(query):
    query = preprocess_input_text([query])
    return any(keyword in query[0] for keyword in symptom_keywords)

# get response from the NLP model
def get_NLPModel_response(user_input):
    processed_text = preprocess_input_text([user_input])
    vectorized_text = vectorizer.transform(processed_text)
    predicted_label = nlp_model.predict(vectorized_text)
    return predicted_label[0]

def get_ClassificationModel_response(Symptom_array):
    input_data = np.array(Symptom_array).reshape(1, -1)
    y_pred = classifier_model.predict(input_data)
    y_proba = classifier_model.predict_proba(input_data)
    confidence = np.max(y_proba)

    if confidence == 1.0:
        return y_pred[0]
    else:
        return None

# Get the response from the Gemini API
def get_final_response(predicted_label):
    headers = {
        'Content-Type': 'application/json',
    }

    user_input = f"Predicted label: {predicted_label}. Give some home remedies and consulting to cure this disease and nothing else please. and dont use negative words like `I cant` or `I dont` or `I wont` or `I will not` or `I will never` or `I will not` or `I will not be able to` or `I will not be able` or `I will not be`."

    data = {
        "contents": [{
            "parts": [{"text": user_input}]
        }]
    }

    try:
        response = requests.post(f"{api_url}?key={api_key}", headers=headers, data=json.dumps(data))
        response.raise_for_status()
        api_response = response.json()

        if 'candidates' in api_response and len(api_response['candidates']) > 0:
            return api_response['candidates'][0]['content']['parts'][0]['text']
        else:
            return "I'm sorry, I didn't get a valid response."

    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting Gemini API: {e}")
        return "I'm sorry, there was an error processing your request."
    
def get_response():
    if 'done_symptom' not in st.session_state:
        st.session_state['done_symptom'] = []

    available_symptoms = [symptom for symptom in symptoms if symptom not in st.session_state['done_symptom']]
    if not available_symptoms:
        st.session_state['done_symptom'] = []
        available_symptoms = symptoms

    random_symptom = random.choice(available_symptoms)
    st.session_state['done_symptom'].append(random_symptom)

    phrases = [
        f"Can you please provide more details about your symptom? For example, do you have {random_symptom}?",
        f"I'm here to help! Could you tell me if you are experiencing {random_symptom}?",
        f"To assist you better, could you let me know if you have {random_symptom}?",
        f"Please share more about your symptoms. Are you feeling {random_symptom}?",
        f"Could you specify if you are having {random_symptom}?"
    ]
    return random.choice(phrases)

def handle_prompt(prompt):
    if contains_symptom_keywords(prompt):
        Symptom_array = get_NLPModel_response(prompt)
        st.session_state.global_array = [st.session_state.global_array[i] | Symptom_array[i] for i in range(len(st.session_state.global_array))]
        st.session_state.update_counter += 1

        if st.session_state.update_counter == 1:
            st.session_state.global_array = [1 if x == 1 else 0 for x in st.session_state.global_array]
            st.session_state.predicted_label = get_ClassificationModel_response(st.session_state.global_array)
            st.session_state.update_counter = 0

        heading = f"{st.session_state.predicted_label}. \n\n"

        doctor_info = f"\n\nDoctor Specialist in {st.session_state.predicted_label} Treatment:\n\nDr. Ravi Kumar\nSpecialization: Infectious Disease Specialist, {st.session_state.predicted_label} Treatment\nExperience: 10+ years in treating mosquito-borne diseases, specializing in dengue fever management.\nConsultation Fee: ₹1000 (Approx.)\nLocation: Fortis Healthcare, Marine Drive, Mumbai, Maharashtra, India.\nPhone: +91 98234 56789\n\nNearby Hospitals:\n\nBreach Candy Hospital\nAddress: Breach Candy, Mumbai, Maharashtra, India\nContact: +91 22 1234 5678\n\nJaslok Hospital\nAddress: 15, Dr. Deshmukh Marg, Mumbai, Maharashtra, India\nContact: +91 22 6660 1234\n\nKokilaben Dhirubhai Ambani Hospital\nAddress: Andheri West, Mumbai, Maharashtra, India\nContact: +91 22 4260 6000"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if st.session_state.predicted_label:
                future_gemini_response = executor.submit(get_final_response, st.session_state.predicted_label)
                gemini_response = future_gemini_response.result()
                gemini_response = heading + gemini_response + doctor_info
            else:
                future_gemini_response = executor.submit(get_response)
                gemini_response = future_gemini_response.result()

        st.chat_message("SymptomScout").markdown(gemini_response)
        st.session_state.messages.append({'role': 'SymptomScout', 'content': gemini_response})

        if st.session_state.update_counter == 0:
            st.session_state.global_array = [0] * 40
            st.session_state.predicted_label = None
    else:
        st.chat_message("SymptomScout").markdown("Please enter your symptoms so I can assist you better. I am a medical-focused AI here to help!")
        st.session_state.messages.append({'role': 'SymptomScout', 'content': "Please enter your symptoms so I can assist you better. I am a medical-focused AI here to help!"})

st.title("Ask SymptomScout")

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Ask SymptomScout")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    handle_prompt(prompt)
