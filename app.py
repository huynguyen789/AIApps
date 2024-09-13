import streamlit as st
import os
from openai import OpenAI
import time

# Initialize OpenAI client
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

def load_prompt(filename):
    with open(os.path.join('./prompts', filename), 'r') as file:
        return file.read()

def generate_job_description(job_title, additional_requirements):
    main_prompt = load_prompt('job_description.txt')
    
    user_input = f"Job title: {job_title}. {additional_requirements}"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": main_prompt},
            {"role": "user", "content": user_input}
        ],
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
            
            
def extract_tag(content, tag):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_index = content.find(start_tag) + len(start_tag)
    end_index = content.find(end_tag)
    return content[start_index:end_index].strip()

def main():
    st.set_page_config(page_title="Job Description Generator", page_icon="📝", layout="wide")

    st.title("Job Description Generator 📝")

    st.write("Welcome to the Job Description Generator! Enter the job title and any additional requirements to generate a high-quality job description.")

    job_title = st.text_input("Enter the job title:", placeholder="e.g., Senior Software Engineer")
    additional_requirements = st.text_area("Enter any additional requirements (optional):", 
                                           placeholder="e.g., TS clearance, 5+ years of experience in Python, knowledge of machine learning")

    if st.button("Generate Job Description"):
        if job_title:
            with st.spinner("Generating job description..."):
                job_description_placeholder = st.empty()
                full_content = ""
                
                for content in generate_job_description(job_title, additional_requirements):
                    full_content += content
                    job_description_placeholder.markdown(full_content)

        else:
            st.warning("Please enter a job title.")

    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses AI to generate high-quality job descriptions based on the provided job title and additional requirements. "
        "It's designed to help hiring managers and recruiters create professional job postings quickly and easily."
    )

if __name__ == "__main__":
    main()