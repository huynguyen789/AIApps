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

def generate_job_description(job_title, additional_requirements, prompt):
    user_input = f"Job title: {job_title}. {additional_requirements}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        stream=True,
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

def improve_job_description(original_jd, feedback, job_title, additional_requirements):
    improve_prompt = load_prompt('improve_job_description.txt')
    improve_input = f"""
Original Job Title: {job_title}
Additional Requirements: {additional_requirements}

Original Job Description:
{original_jd}

User Feedback:
{feedback}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": improve_prompt},
            {"role": "user", "content": improve_input}
        ],
        stream=True,
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

def main():
    st.set_page_config(page_title="Job Description Generator", page_icon="📝", layout="wide")
    st.title("Job Description Generator 📝")
    st.write("Welcome to the Job Description Generator! Enter the job title and any additional requirements to generate a high-quality job description.")

    job_title = st.text_input("Enter the job title:", placeholder="e.g., Senior Software Engineer")
    additional_requirements = st.text_area("Enter any additional requirements (optional):",
                                           placeholder="e.g., TS clearance, 5+ years of experience in Python, knowledge of machine learning")

    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""

    if st.button("Generate Job Description"):
        if job_title:
            with st.spinner("Generating job description..."):
                main_prompt = load_prompt('job_description.txt')
                job_description_placeholder = st.empty()
                full_content = ""
                for content in generate_job_description(job_title, additional_requirements, main_prompt):
                    full_content += content
                    job_description_placeholder.markdown(full_content)
                st.session_state.job_description = full_content
        else:
            st.warning("Please enter a job title.")

    if st.session_state.job_description:
        # st.subheader("Generated Job Description")
        # st.markdown(st.session_state.job_description)

        feedback = st.text_area("Provide feedback to improve the job description:", placeholder="What would you like to change or improve?")
        
        if st.button("Improve Job Description"):
            if feedback:
                with st.spinner("Improving job description..."):
                    improved_jd_placeholder = st.empty()
                    improved_content = ""
                    for content in improve_job_description(st.session_state.job_description, feedback, job_title, additional_requirements):
                        improved_content += content
                        improved_jd_placeholder.markdown(improved_content)
                    st.session_state.job_description = improved_content
            else:
                st.warning("Please provide feedback to improve the job description.")

    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses AI to generate high-quality job descriptions based on the provided job title and additional requirements. "
        "It's designed to help hiring managers and recruiters create professional job postings quickly and easily. "
        "You can also provide feedback to improve the generated job descriptions."
    )

if __name__ == "__main__":
    main()