import os
import re
import pdfplumber
import docx
import spacy
import torch
from transformers import BertTokenizer, BertModel
import logging
from collections import defaultdict
from PIL import Image
import pytesseract
import csv

# Initialize models
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Print to console
)

# Utility functions for text extraction
def extract_phone_numbers(text):
    phone_pattern = re.compile(r'\+?[0-9]{1,4}?[-.\s\(\)]?(\(?\d{1,3}?\)?[-.\s]?)?(\d{1,4}[-.\s]?)?\d{1,4}[-.\s]?\d{1,9}')
    return re.findall(phone_pattern, text)

def extract_emails(text):
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    return re.findall(email_pattern, text)

def extract_skills(text):
    skills_keywords = ["Python", "Java", "JavaScript", "SQL", "Machine Learning", "AI", "Data Science", "Git", "Django", "Flask", "C++", "AWS", "Docker"]
    skills_found = [skill for skill in skills_keywords if skill.lower() in text.lower()]
    return skills_found if skills_found else ["N/A"]

def extract_education(text):
    education_patterns = [
        r'(Bachelor|Master|PhD|Doctorate|B\.?S\.?|M\.?S\.?|M\.?B\.?A\.?)\s+in\s+([a-zA-Z\s]+)',
        r'(Degree|Graduation)\s+in\s+([a-zA-Z\s]+)',
        r'([a-zA-Z\s]+)\s+(Degree|Graduate)'
    ]
    education_matches = []
    for pattern in education_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        education_matches.extend(matches)
    return [match[1] for match in education_matches] or ["N/A"]

def extract_work_experience(text):
    work_exp_patterns = [
        r'(?:Work\s*Experience|Professional\s*Experience|Employment\s*History).*?(?:\n\n|\Z)',
        r'([\w\s]+)\s*(?:at|@)\s*([\w\s]+)\s*(?:from|for)\s*(\d{4}(?:-\d{4})?)'
    ]
    experiences = []
    for pattern in work_exp_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        experiences.extend(matches)
    return experiences if experiences else ["N/A"]

def extract_comprehensive_resume_info(text):
    name = extract_name(text)
    contact_info = extract_contact_info(text)
    skills = extract_skills(text)
    education = extract_education(text)
    work_experience = extract_work_experience(text)

    return {
        "Name": name,
        "Phone": contact_info["phone_numbers"],
        "Email": contact_info["emails"],
        "Skills": ", ".join(skills),
        "Education": ", ".join(education),
        "Work Experience": str(work_experience),
        "Full Text": text
    }

def extract_name(text):
    doc = nlp(text)
    possible_names = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            possible_names.append(ent.text)
    return max(possible_names, key=len) if possible_names else "Name Not Found"

def extract_contact_info(text):
    phone_pattern = r'\+?\d{10,15}'
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_numbers = re.findall(phone_pattern, text)
    emails = re.findall(email_pattern, text)
    return {
        "phone_numbers": ", ".join(phone_numbers) if phone_numbers else "N/A",
        "emails": ", ".join(emails) if emails else "N/A",
    }

# PDF, DOCX, and TXT file extraction
def process_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
                else:
                    image = page.to_image()
                    ocr_text = pytesseract.image_to_string(image.original)
                    text += ocr_text
            return text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        return text.strip()
    except Exception as e:
        return f"Error processing DOCX: {str(e)}"

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error processing TXT: {str(e)}"

def process_documents(documents):
    extracted_data = defaultdict(list)
    
    for file in documents:
        text = ""
        if file.endswith(".pdf"):
            text = process_pdf(file)
        elif file.endswith(".docx"):
            text = extract_text_from_docx(file)
        elif file.endswith(".txt"):
            text = extract_text_from_txt(file)
        
        if text:
            logging.info(f"Processing {file} ...")
            extracted_info = extract_comprehensive_resume_info(text)
            extracted_data[file].append(extracted_info)
        else:
            logging.warning(f"Skipped empty or unprocessable file: {file}")
            
    return extracted_data

# Save extracted data to CSV
def save_to_csv(extracted_data, output_file="extracted_resume_data.csv"):
    fieldnames = ["File Name", "Name", "Phone", "Email", "Skills", "Education", "Work Experience", "Full Text"]
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for file_name, data in extracted_data.items():
            for item in data:
                item["File Name"] = file_name
                writer.writerow(item)

# Main execution with dynamic directory scanning
if __name__ == "__main__":
    # Specify the directory containing resumes
    directory = "./resumes"  # Update to your directory path
    
    # Get all files with supported formats in the directory
    supported_formats = (".pdf", ".docx", ".txt")
    resume_files = [
        os.path.join(directory, file) for file in os.listdir(directory) 
        if file.endswith(supported_formats)
    ]
    
    if not resume_files:
        logging.warning("No resume files found in the specified directory.")
    else:
        extracted_data = process_documents(resume_files)
        save_to_csv(extracted_data)
        logging.info("Extraction complete. Data saved to extracted_resume_data.csv.")
