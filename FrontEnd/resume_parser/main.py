import re
import pdfplumber

import pytesseract
from PIL import Image
import os
import spacy
import smtplib
from email.mime.text import MIMEText
import glob
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from geopy.geocoders import Nominatim

# Set the path for Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\academytraining\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Skill improvement and role mapping (as an example

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def calculate_similarity(text1, text2):
    emb1 = get_bert_embedding(text1)
    emb2 = get_bert_embedding(text2)
    similarity = cosine_similarity(emb1.numpy(), emb2.numpy())
    return similarity[0][0]

def extract_phone_numbers(text):
    phone_patterns = [
        r'\b\d{10}\b',  # Matches exactly 10 digits
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Matches numbers like 638-186-6577
        r'\(\d{3}\)[\s\-.]?\d{3}[\s\-.]?\d{4}',  # Matches numbers like (638) 186-6577
        r'\+91[\s\-.]?\d{4}[\s\-.]?\d{3}[\s\-.]?\d{3}',  # Matches Indian numbers like +91 6380 293 207
    ]
    phone_numbers = []
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            cleaned_match = match.strip()
            cleaned_match_digits = re.sub(r'\D', '', cleaned_match)
            if len(cleaned_match_digits) == 10:
                phone_numbers.append(cleaned_match)
    return list(set(phone_numbers))
import spacy
import re

# Load spaCy's NER model
nlp = spacy.load("en_core_web_sm")

# A more flexible regex to capture phone numbers in different formats
phone_number_regex = r"(\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4})"

# Function to extract the location after the phone number

# Example usage
def is_job_role(line):
    job_roles = ["Software Engineer", "Data Scientist", "Project Manager"]
    return any(role.lower() in line.lower() for role in job_roles)

def is_common_skill(line):
    common_skills = ["JavaScript", "Python", "Java", "C++", "HTML", "CSS"]
    return any(skill.lower() in line.lower() for skill in common_skills)

def extract_name_from_text(text):
    if text is None:
        return None

    name_patterns = [
        re.compile(r'^[A-Z][a-zA-Z\s\-\.]+$', re.MULTILINE),
        re.compile(r'\b[A-Z][a-zA-Z\s\.\-]+\s+[A-Z][a-zA-Z\s\.\-]+\b', re.MULTILINE),
        re.compile(r'\b[A-Z]+\s+[A-Z]+\.[A-Z]+\b', re.IGNORECASE)
    ]

    lines = text.split('\n')
    for line in lines:
        if re.search(r'\d', line) or 'Street' in line or 'Avenue' in line or 'Road' in line:
            continue
        for pattern in name_patterns:
            match = pattern.search(line)
            if match and not is_job_role(line) and not is_common_skill(line):
                return match.group(0).strip()

    return "Name not found"

def extract_emails(text):
    email_pattern = r'([a-zA-Z0-9._%+-]+)(?:\s*|\s*)(@)(?:\s*|\s*)([a-zA-Z0-9.-]+)(?:\s*|\s*)(\.[a-zA-Z]{2,})'
    matches = re.findall(email_pattern, text)
    emails = [f"{username}{at}{domain}{tld}" for username, at, domain, tld in matches]
    return [email.replace(" ", "") for email in emails]

def extract_skills_from_job_description(job_description_text):
    return extract_skills(job_description_text)


# Load spaCy's model
nlp = spacy.load("en_core_web_sm")


def process_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()  # First try extracting text directly
                if not page_text:  # If no text is found, fallback to OCR
                    image = page.to_image()
                    page_text = pytesseract.image_to_string(image.original)
                text += page_text + "\n"
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return text



# A sample list of skills for demonstration; you can enhance this with a more extensive list.
known_skills = [
    "Python", "Java", "C++", "SQL", "Data Science", "Machine Learning", "Deep Learning", "TensorFlow",
    "Keras", "Pandas", "NumPy", "R", "Tableau", "Power BI", "Excel", "Hadoop", "Spark", "AWS", "Azure", "GCP"
]

# Function to extract skills from text
def extract_skills(text):
    # Convert the text to lowercase for case-insensitive matching
    text = text.lower()

    # Find and return skills that are mentioned in the known skills list
    skills_found = [skill for skill in known_skills if skill.lower() in text]
    return skills_found

def extract_skills_from_job_description(job_description_text):
    return extract_skills(job_description_text)

def match_skills(extracted_skills, job_skills):

    # Normalize the skills by converting them to lowercase
    extracted_skills = [skill.lower() for skill in extracted_skills]
    job_skills = [skill.lower() for skill in job_skills]
    
    # Find the common skills between the extracted skills and job description skills
    matched_skills = list(set(extracted_skills) & set(job_skills))
    
    # Calculate the match percentage (based on the number of matched skills)
    if len(job_skills) == 0:
        match_percentage = 0  # If there are no job skills, return 0%
    else:
        match_percentage = (len(matched_skills) / len(job_skills)) * 100
    
    return matched_skills, match_percentage

import re

def extract_location_from_text(text):
    # Example pattern for extracting location based on common city/state/country formats
    location_patterns = [
        r"\b(?:city|state|country|located|from)\s*[:\-]?\s*(\w+|\w+\s\w+)\b",  # Matches 'city:', 'located: New York', etc.
        r"\b(\w{2,},\s*\w{2,})\b",  # Matches city, state, or country combinations like "New York, USA"
        r"\b(?:located\s*in\s*)([A-Za-z\s]+)\b"  # Matches 'located in New York'
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return "Location not found"


def extract_resume_details(resume_path):
    file_extension = os.path.splitext(resume_path)[1].lower()

    # Process the resume text based on file type
    if file_extension == ".pdf":
        text = process_pdf(resume_path)
    else:
        return None  # If it's not a PDF, return None or handle accordingly

    # Extracting name, emails, skills, and phone numbers
    name = extract_name_from_text(text)
    emails = extract_emails(text)
    skills = extract_skills(text)  # Make sure this function is defined
    phone_numbers = extract_phone_numbers(text)

    # Extract location from resume or set default
    location_name = extract_location_from_text(text)  # Extract location from the resume text
    location = extract_location_from_text(location_name)  # Make sure this function is defined
    
    # Return the extracted details as a dictionary
    return {
        "Name": name,
        "Emails": emails,
        "Skills": skills,
        "Phone Numbers": phone_numbers,
        "Location": location
    }


def extract_resumes_from_directory(directory_path):
    # Get a list of all PDF files in the directory
    resume_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.lower().endswith('.pdf')]
    return resume_files

# Function to process resumes
import pandas as pd

def process_resumes(resume_directory, job_description_text, threshold=50, resumes_to_scan=None, ranked_candidates_to_generate=None):
    resume_files = extract_resumes_from_directory(resume_directory)
    job_skills = extract_skills_from_job_description(job_description_text)
    all_results = []
    ranked_candidates = []

    # Initialize counters for scanned resumes
    scanned_resumes_count = 0

    # Iterate over uploaded resumes and process each
    for resume_path in resume_files:
        if resumes_to_scan and scanned_resumes_count >= resumes_to_scan:
            break  # Stop scanning once the desired number of resumes is reached

        print(f"Screening resume: {resume_path}")

        resume_details = extract_resume_details(resume_path)
        if resume_details:
            candidate_name = resume_details['Name']
            extracted_skills = resume_details['Skills']
            recipient_email = resume_details['Emails'][0] if resume_details['Emails'] else None
            phone_numbers = resume_details['Phone Numbers']
            location = resume_details['Location']

            # Compare extracted skills with job description skills
            matched_skills, match_percentage = match_skills(extracted_skills, job_skills)
            
            # Determine status
            status = "Selected" if match_percentage >= threshold else "Not Selected"
            
            # Compile the result for this resume
            result_entry = {
                'Candidate Name': candidate_name,
                'Phone Numbers': ', '.join(phone_numbers),
                'Emails': ', '.join(resume_details['Emails']),
                'Location': location,
                'Skills Matched': ', '.join(matched_skills),
                'Match Percentage': match_percentage,
                'Status': status,
            }

            all_results.append(result_entry)

            # Add to ranked candidates list for sorting later
            ranked_candidates.append(result_entry)

            scanned_resumes_count += 1

    # Sort ranked candidates based on match percentage in descending order
    ranked_candidates = sorted(ranked_candidates, key=lambda x: x['Match Percentage'], reverse=True)

    # If a number of ranked candidates is specified, limit the list
    if ranked_candidates_to_generate:
        ranked_candidates = ranked_candidates[:ranked_candidates_to_generate]

    # Convert results to DataFrame
    all_results_df = pd.DataFrame(all_results)
    ranked_candidates_df = pd.DataFrame(ranked_candidates)

    # Save the DataFrames to CSV files
    all_resumes_file_path = "all_resumes.csv"
    ranked_candidates_file_path = "ranked_candidates.csv"

    all_results_df.to_csv(all_resumes_file_path, index=False)
    ranked_candidates_df.to_csv(ranked_candidates_file_path, index=False)

    # Optionally, you can print the file paths or return them
    print(f"All resumes saved to: {all_resumes_file_path}")
    print(f"Ranked candidates saved to: {ranked_candidates_file_path}")

    return all_resumes_file_path, ranked_candidates_file_path

# Example usage:
resume_directory = r"D:\FrontEnd\resume_parser\uploads"  # Replace with your directory path
job_description_text = input("Enter your job description: ")
threshold = float(input("Enter the minimum match threshold (0-100): "))  # Take user input for threshold
resumes_to_scan = int(input("Enter the number of resumes to scan: "))  # Take user input for number of resumes to scan
ranked_candidates_to_generate = int(input("Enter the number of ranked candidates to generate: "))  # Take user input for ranked candidates

all_resumes_file, ranked_candidates_file = process_resumes(
    resume_directory, job_description_text, threshold, resumes_to_scan, ranked_candidates_to_generate
)


# Example usage:
resume_directory = r"D:\FrontEnd\resume_parser\uploads"  # Replace with your directory path
job_description_text = input("Enter your job description: ")
threshold = float(input("Enter the minimum match threshold (0-100): "))  # Take user input for threshold

all_results_file, selected_results_file, not_selected_results_file = process_resumes(resume_directory, job_description_text, threshold)

# Now you can use the in-memory files as 

# Example usage:
import os

# Replace with the actual path

# Check if the directory exists
if not os.path.exists(resume_directory):
    print(f"Error: Directory '{resume_directory}' does not exist.")
else:
    job_description_text = input("Enter your job description: ").strip()
    threshold = input("Enter the minimum match threshold (0-100, default=50): ").strip()
    threshold = float(threshold) if threshold.isdigit() else 50.0
    resumes_to_scan = input("Enter the number of resumes to scan (default=all): ").strip()
    resumes_to_scan = int(resumes_to_scan) if resumes_to_scan.isdigit() else None

    # Call the process function
    all_resumes_file_path, ranked_candidates_file_path = process_resumes(
        resume_directory, 
        job_description_text, 
        threshold=threshold, 
        resumes_to_scan=resumes_to_scan
    )

    print(f"All resumes processed. Results saved to:\n"
          f"- All resumes: {all_resumes_file_path}\n"
          f"- Ranked candidates: {ranked_candidates_file_path}")

import re
import spacy
from geopy.geocoders import Nominatim

# Load spaCy's NER model
nlp = spacy.load("en_core_web_sm")

# Initialize Geolocator
geolocator = Nominatim(user_agent="location_extractor")

# Predefined phrases and contexts that might indicate the candidate's current location
location_indicators = [
    "currently in", "based in", "residing in", "living in", "from", "hails from",
    "located in", "native of", "staying in", "city of", "region of", "working from"
]

# Function to extract locations from the text using NER and contextual filtering
def extract_location_from_text(text):
    """
    Extracts a location from the provided text using spaCy's NER model
    and validates it with Geopy.
    """
    doc = nlp(text)
    location = None

    # Loop through all the named entities recognized by spaCy
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:  # GPE: Geopolitical entity, LOC: Location
            location = ent.text.strip()  # Get the recognized location entity
            break

    # If no location was found, use regex patterns for location indicators
    if not location:
        location = find_location_using_regex(text)

    # If a location was found, try to get more details using geopy
    if location:
        try:
            # Attempt to geolocate the found location
            location_info = geolocator.geocode(location, timeout=10)
            if location_info:
                # Return the address or just the city/state
                return location_info.address.split(",")[0]  # City or state
            else:
                return f"Location lookup failed for '{location}'"
        except Exception as e:
            return f"Error during location lookup: {e}"

    return "Location not found"

def find_location_using_regex(text):
    """
    Searches for location-like patterns (such as city names, countries, or regions)
    using regular expressions.
    """
    # Predefined patterns for detecting location names
    location_patterns = [
        r"\b(?:New York|Los Angeles|San Francisco|London|Paris|Berlin|Tokyo|Sydney|Delhi)\b",  # Example cities
        r"\b(?:USA|Canada|India|UK|Australia|Germany|France)\b",  # Example countries
        r"\b(?:California|Texas|Florida|Ontario|Maharashtra)\b"  # Example states/provinces
    ]

    # Search the text for location patterns
    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return None  # Return None if no location was found

def extract_location_from_indicators(text):
    """
    Attempts to extract the location from phrases like 'currently in', 'based in', etc.
    """
    # Use regex to find location indicators
    for indicator in location_indicators:
        match = re.search(rf"{indicator}\s+([A-Za-z\s]+)", text, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            # Validate the location using Geopy
            location_info = geolocator.geocode(location, timeout=10)
            if location_info:
                return location_info.address.split(",")[0]  # City or state
            else:
                return f"Location lookup failed for '{location}'"
    
    return "Location not found"

def validate_location(location: str) -> bool:
    """
    Validates the location by checking if it's a valid city or region using Geopy.
    """
    try:
        # Geolocate the location (checking for a valid city or place)
        location_info = geolocator.geocode(location, timeout=10)
        if location_info:
            # Return True if a valid location is found
            return True
    except Exception as e:
        print(f"Error validating location {location}: {e}")
    return False

def extract_candidate_location(text):
    """
    Extracts the candidate's location from the resume text using a combination of spaCy, regex, and geopy.
    """
    # First attempt to extract location using spaCy NER model
    location = extract_location_from_text(text)

    # If no location is found from spaCy, attempt using location indicators
    if location == "Location not found":
        location = extract_location_from_indicators(text)

    # Return the extracted location
    return location

# Example usage:

