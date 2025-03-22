import spacy
from keybert import KeyBERT

def extract_job_details(job_description):
    nlp = spacy.load("en_core_web_sm")
    kw_model = KeyBERT()
    
    # Process the job description with spaCy
    doc = nlp(job_description)
    
    # Extract named entities related to responsibilities, qualifications, and skills
    responsibilities = []
    qualifications = []
    skills = []
    
    for sent in doc.sents:
        if any(word in sent.text.lower() for word in ["responsible", "manage", "oversee", "develop", "coordinate", "execute"]):
            responsibilities.append(sent.text.strip())
        elif any(word in sent.text.lower() for word in ["must have", "required", "qualification", "degree", "experience"]):
            qualifications.append(sent.text.strip())
    
    # Extract key skills using KeyBERT
    extracted_keywords = kw_model.extract_keywords(job_description, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    skills = [keyword[0] for keyword in extracted_keywords]
    
    return {
        "Responsibilities": responsibilities,
        "Qualifications": qualifications,
        "Skills": skills
    }

if __name__ == "__main__":
    job_desc = """We are looking for an experienced Software Engineer to develop and maintain scalable applications. 
    The candidate will be responsible for designing, coding, and reviewing software solutions.
    Must have a Bachelor's degree in Computer Science or related field with at least 3 years of experience in Python and backend development.
    Required skills include Python, Django, REST APIs, and cloud platforms like AWS or Azure."""
    
    job_details = extract_job_details(job_desc)
    
    print("Responsibilities:")
    print("\n".join(job_details["Responsibilities"]))
    print("\nQualifications:")
    print("\n".join(job_details["Qualifications"]))
    print("\nSkills:")
    print(", ".join(job_details["Skills"]))
