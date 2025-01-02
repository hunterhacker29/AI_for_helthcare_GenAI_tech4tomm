

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''


# import streamlit as st
# import pickle
# import re
# import requests
# import json
# import concurrent.futures

# with open('logistic_regression_model.pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)

# with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
#     loaded_tfidf = pickle.load(tfidf_file)

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
#     new_text_tfidf = loaded_tfidf.transform(processed_text)
    
#     predicted_label = loaded_model.predict(new_text_tfidf)
    
    
#     return predicted_label[0]

# def get_gemini_response(predicted_label):
#     API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
#     API_KEY = "AIzaSyAWfOaA_zYrilK7uQeuC3Mh0552PaoMRzo"  # Replace with your actual API key

#     headers = {
#         'Content-Type': 'application/json',
#     }

#     user_input = f"Predicted label: {predicted_label}. Give some info about this and suggest remedies, home remedies to cure this, and provide recommendations on managing it."

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
#         "blood in sputum", "persistent cough", "loss of sensation", "blurry vision", "irritability", "delayed growth"
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
#         # Use ThreadPoolExecutor to run both tasks in parallel
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             # Run the model prediction and Gemini API call in parallel
#             future_label = executor.submit(get_model_response, prompt)
#             predicted_label = future_label.result()  # This will block until the prediction is done
            
#             # Feed the predicted label to the API for information, remedies, and recommendations
#             future_gemini_response = executor.submit(get_gemini_response, predicted_label)
#             gemini_response = future_gemini_response.result()  # This will block until the response is ready
        
#         st.chat_message("SymptomScout").markdown(f"Predicted label: {predicted_label}")
#         st.session_state.messages.append({'role': 'SymptomScout', 'content': f"Predicted label: {predicted_label}"})
        
#         st.chat_message("SymptomScout").markdown(gemini_response)
#         st.session_state.messages.append({'role': 'SymptomScout', 'content': gemini_response})
#     else:
        
#         st.chat_message("SymptomScout").markdown("Please enter your symptoms so I can assist you better. I am a medical-focused AI here to help!")
#         st.session_state.messages.append({'role': 'SymptomScout', 'content': "Please enter your symptoms so I can assist you better. I am a medical-focused AI here to help!"})




''''''''''''''''''''''''''''''''''''''''''''''''''

import streamlit as st
import pickle
import re
import requests
import json
import os

# Define raw file URLs
model_url = "https://raw.githubusercontent.com/hunterhacker29/AI_for_helthcare_GenAI_tech4tomm/main/logistic_regression_model.pkl"
tfidf_url = "https://raw.githubusercontent.com/hunterhacker29/AI_for_helthcare_GenAI_tech4tomm/main/tfidf_vectorizer.pkl"

# Function to download the models
def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        st.error(f"Error downloading file from {url}. Status code: {response.status_code}")

# Function to load models (only once at the start)
def load_models():
    # Download the models if they are not already present
    if not os.path.exists('logistic_regression_model.pkl'):
        download_file(model_url, 'logistic_regression_model.pkl')
    if not os.path.exists('tfidf_vectorizer.pkl'):
        download_file(tfidf_url, 'tfidf_vectorizer.pkl')

    # Load the models from the local file system after downloading
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
        loaded_tfidf = pickle.load(tfidf_file)

    return loaded_model, loaded_tfidf

# Load the pre-trained models
loaded_model, loaded_tfidf = load_models()

# Proceed if models are loaded successfully
if not loaded_model or not loaded_tfidf:
    st.stop()

def preprocess_input_text(input_text):
    processed_text = []
    for text in input_text:
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        processed_text.append(text)
    return processed_text

def get_model_response(user_input):
    processed_text = preprocess_input_text([user_input])
    new_text_tfidf = loaded_tfidf.transform(processed_text)
    
    predicted_label = loaded_model.predict(new_text_tfidf)
    return predicted_label[0]

def get_gemini_response(predicted_label):
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    API_KEY = "AIzaSyAWfOaA_zYrilK7uQeuC3Mh0552PaoMRzo"  # Replace with your actual API key

    headers = {
        'Content-Type': 'application/json',
    }

    user_input = f"Predicted label: {predicted_label}. Give some info about this and suggest remedies, home remedies to cure this, and provide recommendations on managing it."

    data = {
        "contents": [{
            "parts": [{"text": user_input}]
        }]
    }

    try:
        # Send the request to the Gemini API
        response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an error if the request fails
        api_response = response.json()  # Parse the JSON response

        if 'candidates' in api_response and len(api_response['candidates']) > 0:
            return api_response['candidates'][0]['content']['parts'][0]['text']
        else:
            return "I'm sorry, I didn't get a valid response."
    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting Gemini API: {e}")
        return "I'm sorry, there was an error processing your request."

# Function to check if the input contains symptom-related terms
def contains_symptom_keywords(query):
    symptom_keywords = [
        "thirst", "urination", "fatigue", "weight loss", "vision", "wounds", "healing", "blurred vision", "slow-healing",
        "headache", "fever", "joint pain", "rash", "nausea", "vomiting", "muscle pain", "cough", "chest pain", 
        "abdominal pain", "shortness of breath", "swelling", "dizziness", "loss of appetite", "diarrhea", "jaundice",
        "dark urine", "skin lesions", "numbness", "pain during intercourse", "pelvic pain", "leg cramps", "hydrophobia",
        "swollen lymph nodes", "wheezing", "difficulty swallowing", "hoarseness", "sore throat", "stiff neck", 
        "body aches", "irregular heartbeat", "diabetic", "immune system", "contagious", "muscle weakness", "weight gain",
        "blood in sputum", "persistent cough", "loss of sensation", "blurry vision", "irritability", "delayed growth"
    ]
    
    query = query.lower()
    return any(keyword in query for keyword in symptom_keywords)

# Streamlit app code
st.title("Ask SymptomScout")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for messages in st.session_state.messages:
    st.chat_message(messages['role']).markdown(messages['content'])

# User input
prompt = st.chat_input("Ask SymptomScout")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    if contains_symptom_keywords(prompt):  # Check if the input contains symptoms
        # Get model response
        predicted_label = get_model_response(prompt)
        
        # Get Gemini API response
        gemini_response = get_gemini_response(predicted_label)
        
        st.chat_message("SymptomScout").markdown(f"Predicted label: {predicted_label}")
        st.session_state.messages.append({'role': 'SymptomScout', 'content': f"Predicted label: {predicted_label}"})
        
        st.chat_message("SymptomScout").markdown(gemini_response)
        st.session_state.messages.append({'role': 'SymptomScout', 'content': gemini_response})
    else:
        st.chat_message("SymptomScout").markdown("Please enter your symptoms so I can assist you better. I am a medical-focused AI here to help!")
        st.session_state.messages.append({'role': 'SymptomScout', 'content': "Please enter your symptoms so I can assist you better. I am a medical-focused AI here to help!"})
