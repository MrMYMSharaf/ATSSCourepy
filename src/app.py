import streamlit as st
from pdfminer.high_level import extract_text
import pandas as pd
import os
import re
import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
import docx2txt  # Added for Word document support

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Gemini API
if not api_key:
    raise ValueError("Google API key is missing. Please set it in the .env file.")

# Updated model name and configuration
llm = GoogleGenerativeAI(
    model="gemini-1.5-pro",  # Updated model name
    google_api_key=api_key,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

# Function to extract text from Word document
def extract_text_from_docx(docx_path):
    return docx2txt.process(docx_path)

# Function to extract text based on file type
def extract_text_from_file(file_path, file_type):
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == "docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# Function to extract JSON from text that might contain markdown or other formatting
def extract_json_from_text(text):
    # Try to find JSON content within markdown code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    match = re.search(json_pattern, text)
    
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON content without markdown
    try:
        # Try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find any JSON-like structure
        curly_brace_pattern = r'(\{[\s\S]*\})'
        match = re.search(curly_brace_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    
    # If all attempts fail, return the original text in a structured format
    return {
        "Error": "Failed to parse JSON",
        "Raw": text[:1000] if len(text) > 1000 else text
    }

# Function to extract key information using Google Gemini AI
def extract_info(text, job_requirements):
    prompt = f"""
    You are a professional resume analyzer. Extract key details from the following resume text and format your response STRICTLY as valid JSON:
    
    RESUME TEXT:
    {text}
    
    JOB REQUIREMENTS:
    {job_requirements}
    
    Return ONLY a properly formatted JSON object with these exact fields (include null for missing information):
    {{
      "Name": "Full name of candidate",
      "Age": null or age if present,
      "Phone Number": "Phone number if present",
      "Email": "Email address",
      "Address": "Address if present or null",
      "Working History": [
        {{
          "Company": "Company name",
          "Title": "Job title",
          "Dates": "Employment dates",
          "Responsibilities": ["Key responsibility 1", "Key responsibility 2"]
        }}
      ],
      "Education": [
        {{
          "Institution": "School/University name",
          "Degree": "Degree obtained",
          "Dates": "Study dates"
        }}
      ],
      "Skills": ["Skill 1", "Skill 2", "Skill 3"],
      "AI Analyst Overview": "Brief summary of candidate qualifications",
      "AI Analyst Top 3 Strengths": ["Strength 1", "Strength 2", "Strength 3"],
      "AI Analyst Top 3 Weaknesses": ["Weakness 1", "Weakness 2", "Weakness 3"],
      "Background Verification Score": number between 1-10,
      "ATS Score": number between 1-10,
      "Job Fit Score": number between 1-10,
      "Overall Suitability Score": number between 1-100
    }}
    
    IMPORTANT: For the Overall Suitability Score, calculate a weighted score from 1-100 based on:
    - Job Fit Score (50% weight)
    - ATS Score (30% weight)
    - Background Verification Score (20% weight)
    
    Your response MUST be properly formatted JSON ONLY with no other text before or after the JSON.
    """
    
    try:
        response = llm.invoke(prompt)
        # Process the response to extract JSON
        return extract_json_from_text(response)
    except Exception as e:
        return {"Error": str(e)}

# Flatten nested dictionaries for CSV export
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if all(isinstance(i, dict) for i in v if i):
                # Process list of dictionaries (like work history)
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(flatten_dict(item, f"{new_key}_{i+1}", sep=sep).items())
            else:
                # Join list items with commas
                items.append((new_key, ", ".join(str(x) for x in v if x)))
        else:
            items.append((new_key, v))
    return dict(items)

# Calculate overall score for ranking (fallback if AI doesn't provide it)
def calculate_overall_score(result):
    try:
        # Check if Overall Suitability Score is already provided
        if "Overall Suitability Score" in result and isinstance(result["Overall Suitability Score"], (int, float)):
            return result["Overall Suitability Score"]
        
        # Otherwise calculate it from component scores
        job_fit = float(result.get("Job Fit Score", 0))
        ats = float(result.get("ATS Score", 0))
        background = float(result.get("Background Verification Score", 0))
        
        # Calculate weighted score (50% Job Fit, 30% ATS, 20% Background)
        return (job_fit * 5) + (ats * 3) + (background * 2)
    except (ValueError, TypeError):
        return 0

# Streamlit App
def main():
    st.title("Resume Analyzer & Ranking Application")
    st.write("Analyze multiple resumes, rank them by suitability, and download the results as a CSV file.")
    
    with st.sidebar:
        st.header("Upload Resumes")
        uploaded_files = st.file_uploader("Upload Resumes (PDF or Word)", type=["pdf", "docx"], accept_multiple_files=True)
    
    st.header("Job Requirements")
    job_requirements = st.text_area("Paste the job requirements or description here:", height=200)
    
    if st.button("Analyze & Rank Resumes"):
        if not uploaded_files:
            st.warning("Please upload at least one resume.")
        elif not job_requirements:
            st.warning("Please enter the job description.")
        else:
            st.write(f"Total Resumes Uploaded: {len(uploaded_files)}")
            
            progress_bar = st.progress(0)
            results = []
            flattened_results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Analyzing resume {i+1}/{len(uploaded_files)}..."):
                    # Determine file type
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    # Save the uploaded file temporarily
                    temp_filename = f"temp_resume_{i}.{file_extension}"
                    with open(temp_filename, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Extract text from the saved file based on its type
                        text = extract_text_from_file(temp_filename, file_extension)
                        
                        # Extract information
                        info = extract_info(text, job_requirements)
                        
                        # Add filename to results
                        info["Filename"] = uploaded_file.name
                        
                        # Ensure overall score exists
                        if "Overall Suitability Score" not in info or not isinstance(info["Overall Suitability Score"], (int, float)):
                            info["Overall Suitability Score"] = calculate_overall_score(info)
                        
                        # Store original result
                        results.append(info)
                        
                        # Create flattened version for CSV
                        flattened_results.append(flatten_dict(info))
                        
                    except Exception as e:
                        error_result = {
                            "Error": str(e), 
                            "Filename": uploaded_file.name,
                            "Overall Suitability Score": 0
                        }
                        results.append(error_result)
                        flattened_results.append(error_result)
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Rank the results based on the overall score
            ranked_results = sorted(results, key=lambda x: float(x.get("Overall Suitability Score", 0)), reverse=True)
            
            # Add rank to each result
            for i, result in enumerate(ranked_results):
                result["Rank"] = i + 1
                
            # Update the flattened results with ranks
            for flat_result in flattened_results:
                for ranked in ranked_results:
                    if flat_result.get("Filename") == ranked.get("Filename"):
                        flat_result["Rank"] = ranked["Rank"]
                        break
            
            # Display results
            st.subheader("Analysis Results")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Ranking Summary", "Detailed View", "Comparison View", "Raw JSON"])
            
            with tab1:
                try:
                    # Create a simplified dataframe for the ranking view
                    ranking_data = []
                    for r in ranked_results:
                        if "Error" in r:
                            ranking_data.append({
                                "Rank": r.get("Rank", "N/A"),
                                "Filename": r.get("Filename", "Unknown"),
                                "Name": "Error",
                                "Overall Score": 0,
                                "Job Fit": 0,
                                "ATS Score": 0,
                                "Top Skills": "Error processing resume"
                            })
                        else:
                            skills = r.get("Skills", [])
                            if isinstance(skills, list):
                                skills_str = ", ".join(skills[:5]) if len(skills) > 0 else "Not found"
                            else:
                                skills_str = str(skills)
                                
                            ranking_data.append({
                                "Rank": r.get("Rank", "N/A"),
                                "Filename": r.get("Filename", "Unknown"),
                                "Name": r.get("Name", "Unknown"),
                                "Overall Score": r.get("Overall Suitability Score", 0),
                                "Job Fit": r.get("Job Fit Score", 0),
                                "ATS Score": r.get("ATS Score", 0),
                                "Background Score": r.get("Background Verification Score", 0),
                                "Top Skills": skills_str
                            })
                    
                    ranking_df = pd.DataFrame(ranking_data)
                    
                    # Display ranking table without background gradient
                    st.write("### Candidate Ranking")
                    
                    # Format the dataframe for display with custom formatting instead of background gradient
                    # We'll use Streamlit's dataframe highlighting instead
                    st.dataframe(
                        ranking_df,
                        use_container_width=True,
                        column_config={
                            "Overall Score": st.column_config.NumberColumn(
                                "Overall Score",
                                help="Overall suitability score out of 100",
                                format="%.1f/100"
                            ),
                            "Job Fit": st.column_config.NumberColumn(
                                "Job Fit",
                                help="Job fit score out of 10",
                                format="%.1f/10"
                            ),
                            "ATS Score": st.column_config.NumberColumn(
                                "ATS Score",
                                help="ATS score out of 10",
                                format="%.1f/10"
                            ),
                            "Background Score": st.column_config.NumberColumn(
                                "Background Score",
                                help="Background verification score out of 10",
                                format="%.1f/10"
                            )
                        }
                    )
                    
                    # Display top candidates
                    st.write("### Top 3 Candidates")
                    top_candidates = ranking_df.head(3)
                    for i, (_, candidate) in enumerate(top_candidates.iterrows()):
                        st.write(f"**#{i+1}: {candidate['Name']}** - Overall Score: {candidate['Overall Score']}/100")
                        st.write(f"Job Fit: {candidate['Job Fit']}/10 | ATS Score: {candidate['ATS Score']}/10 | Skills: {candidate['Top Skills']}")
                        st.write("---")
                    
                except Exception as e:
                    st.error(f"Error creating ranking: {str(e)}")
            
            with tab2:
                for result in ranked_results:
                    with st.expander(f"Rank #{result.get('Rank', 'N/A')}: {result.get('Filename', 'Unknown')} - {result.get('Name', 'Unknown')} (Score: {result.get('Overall Suitability Score', 0)}/100)"):
                        if "Error" in result:
                            st.error(f"Error: {result['Error']}")
                            if "Raw" in result:
                                with st.expander("Raw Response"):
                                    st.text(result["Raw"])
                        else:
                            # Display formatted resume details
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Personal Information")
                                st.write(f"**Name:** {result.get('Name', 'Not found')}")
                                st.write(f"**Email:** {result.get('Email', 'Not found')}")
                                st.write(f"**Phone:** {result.get('Phone Number', 'Not found')}")
                                st.write(f"**Address:** {result.get('Address', 'Not found')}")
                            
                            with col2:
                                st.subheader("Scores")
                                st.write(f"**Overall Suitability:** {result.get('Overall Suitability Score', 'N/A')}/100")
                                st.write(f"**Job Fit Score:** {result.get('Job Fit Score', 'N/A')}/10")
                                st.write(f"**ATS Score:** {result.get('ATS Score', 'N/A')}/10")
                                st.write(f"**Background Score:** {result.get('Background Verification Score', 'N/A')}/10")
                            
                            st.subheader("Skills")
                            skills = result.get('Skills', [])
                            if isinstance(skills, list):
                                st.write(", ".join(skills))
                            else:
                                st.write(skills)
                            
                            st.subheader("Working History")
                            work_history = result.get('Working History', [])
                            if isinstance(work_history, list):
                                for job in work_history:
                                    if isinstance(job, dict):
                                        st.markdown(f"**{job.get('Title', 'Unknown Position')}** at {job.get('Company', 'Unknown Company')} ({job.get('Dates', 'No dates')})")
                                        responsibilities = job.get('Responsibilities', [])
                                        if isinstance(responsibilities, list):
                                            for resp in responsibilities:
                                                st.markdown(f"- {resp}")
                                        else:
                                            st.write(responsibilities)
                            else:
                                st.write(work_history)
                            
                            st.subheader("Education")
                            education = result.get('Education', [])
                            if isinstance(education, list):
                                for edu in education:
                                    if isinstance(edu, dict):
                                        st.write(f"**{edu.get('Degree', 'Unknown Degree')}** from {edu.get('Institution', 'Unknown Institution')} ({edu.get('Dates', 'No dates')})")
                            else:
                                st.write(education)
                            
                            st.subheader("AI Analysis")
                            st.write(f"**Overview:** {result.get('AI Analyst Overview', 'Not available')}")
                            
                            st.write("**Top 3 Strengths:**")
                            strengths = result.get('AI Analyst Top 3 Strengths', [])
                            if isinstance(strengths, list):
                                for strength in strengths:
                                    st.markdown(f"- {strength}")
                            else:
                                st.write(strengths)
                            
                            st.write("**Areas for Improvement:**")
                            weaknesses = result.get('AI Analyst Top 3 Weaknesses', [])
                            if isinstance(weaknesses, list):
                                for weakness in weaknesses:
                                    st.markdown(f"- {weakness}")
                            else:
                                st.write(weaknesses)
            
            with tab3:
                # Create comparison data for visualization
                comparison_data = []
                for r in ranked_results:
                    if "Error" not in r:
                        comparison_data.append({
                            "Name": r.get('Name', 'Unknown'),
                            "Overall Score": float(r.get('Overall Suitability Score', 0)),
                            "Job Fit": float(r.get('Job Fit Score', 0)),
                            "ATS Score": float(r.get('ATS Score', 0)),
                            "Background Score": float(r.get('Background Verification Score', 0))
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Horizontal bar chart for overall scores
                    st.write("### Overall Candidate Comparison")
                    st.bar_chart(comparison_df.set_index('Name')['Overall Score'], use_container_width=True)
                    
                    # Side-by-side component scores
                    st.write("### Component Scores Comparison")
                    component_comparison = comparison_df.set_index('Name')[['Job Fit', 'ATS Score', 'Background Score']]
                    st.bar_chart(component_comparison, use_container_width=True)
                    
                    # Skill comparison (show top skills of top 5 candidates)
                    st.write("### Skills Comparison (Top 5 Candidates)")
                    top_skills = {}
                    for i, r in enumerate(ranked_results[:5]):
                        if "Error" not in r:
                            skills = r.get('Skills', [])
                            if isinstance(skills, list):
                                top_skills[r.get('Name', f'Candidate {i+1}')] = skills[:5] if len(skills) > 0 else ["No skills found"]
                    
                    # Display skills side by side
                    cols = st.columns(len(top_skills) if len(top_skills) > 0 else 1)
                    for i, (name, skills) in enumerate(top_skills.items()):
                        with cols[i]:
                            st.write(f"**{name}**")
                            for skill in skills:
                                st.write(f"- {skill}")
                
                else:
                    st.write("No valid data for comparison")
            
            with tab4:
                st.json(ranked_results)
            
            # Create downloadable CSV
            try:
                # Sort the flattened results by rank
                sorted_flattened = sorted(flattened_results, key=lambda x: int(x.get("Rank", float('inf'))))
                df = pd.DataFrame(sorted_flattened)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Full Analysis & Ranking (CSV)",
                    data=csv,
                    file_name="resume_analysis_ranked.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error creating CSV: {str(e)}")

if __name__ == "__main__":
    main()