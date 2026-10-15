import os
import glob
from pypdf import PdfReader
from google import genai
from google.genai import types

def extract_text_from_pdfs(input_dir):
    combined_text = ""
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{input_dir}' directory.")
    print(f"Found {len(pdf_files)} PDF files to process...")
    for pdf_path in pdf_files:
        print(f"Reading {pdf_path}...")
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                combined_text += text + "\n"
    return combined_text

def generate_mcq_csv(unit_name="Unit_1_Political_Theory", target_count=50):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing!")

    client = genai.Client(api_key=api_key)
    source_material = extract_text_from_pdfs("inputs")

    prompt = f"""
    You are an expert examiner for the West Bengal State Eligibility Test (WB SET) in Political Science.
    
    Based on the provided study materials and syllabus below, generate EXACTLY {target_count} high-standard multiple-choice questions (MCQs) for {unit_name}.
    
    EXAM PATTERN REQUIREMENTS:
    - Questions must reflect SET Paper-II standards (Statement verification, Assertion-Reason, Thinker-Book matching, and Chronological order).
    - Provide 4 distinct options per question.
    - Provide an authoritative academic explanation for every question.
    
    OUTPUT FORMAT INSTRUCTIONS:
    Return the output STRICTLY in CSV format (using comma separator, properly quoted fields).
    Do NOT include markdown formatting or code blocks like ```csv.
    
    Columns required:
    "Question","Option A","Option B","Option C","Option D","Correct Answer","Explanation"
    
    - "Correct Answer" column MUST contain exactly one of: "Option A", "Option B", "Option C", or "Option D".
    
    CONTEXT MATERIAL:
    {source_material[:40000]}
    """

    print("Sending prompt to Gemini API...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )

    raw_output = response.text.strip()
    if raw_output.startswith("```csv"):
        raw_output = raw_output[6:]
    if raw_output.startswith("```"):
        raw_output = raw_output[3:]
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]
    raw_output = raw_output.strip()

    os.makedirs("outputs", exist_ok=True)
    output_filename = f"outputs/SET_{unit_name.replace(' ', '_')}_Questions.csv"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(raw_output)

    print(f"Successfully generated question bank: {output_filename}")

if __name__ == "__main__":
    unit = os.environ.get("UNIT_NAME", "Unit_1_Political_Theory")
    count = int(os.environ.get("QUESTION_COUNT", "50"))
    generate_mcq_csv(unit_name=unit, target_count=count)
