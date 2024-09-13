import streamlit as st
import os
from openai import OpenAI
import time
import asyncio
import json
import re
from collections import deque
from anthropic import AsyncAnthropic

# Initialize OpenAI client
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# Initialize Anthropic client
anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)

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

async def get_model_answer(instruction, prompt, model_name):
    if "claude" in model_name.lower():
        message = await anthropic_client.messages.create(
            model=model_name,
            max_tokens=3000,
            temperature=0,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": instruction
                }
            ]
        )
        return message.content[0].text
    else:  # OpenAI models
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": instruction}
        ]
        response = await client.chat.completions.acreate(
            model=model_name,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

def remove_analysis(prompt):
    return re.sub(r'<analysis>.*?</analysis>', '', prompt, flags=re.DOTALL)

async def prompt_generator(user_request):
    iteration = 0
    optimizer_model = "claude-3-sonnet-20240229"  # or "gpt-4"
    current_prompt = ""

    while True:
        improvement_prompt = load_prompt('create_adv_prompt.txt')
        
        if iteration == 0:
            improvement_prompt = improvement_prompt.format(
                user_request=user_request,
                user_feedback="No user feedback yet",
                current_prompt="No current prompt yet."
            )
        else:
            improvement_prompt = improvement_prompt.format(
                user_request=user_request,
                user_feedback=f"User feedback: {st.session_state.user_feedback}",
                current_prompt=current_prompt
            )

        improved_prompt = await get_model_answer(improvement_prompt, "", optimizer_model)
        current_prompt = remove_analysis(improved_prompt)
        yield current_prompt
        iteration += 1

        if st.session_state.user_feedback.strip().lower() == 'end':
            break

def main():
    st.set_page_config(page_title="AI Assistant Tools", page_icon="🛠️", layout="wide")

    st.title("AI Assistant Tools 🛠️")

    tool_choice = st.sidebar.radio("Choose a tool:", ("Job Description Generator", "Prompt Generator"))

    if tool_choice == "Job Description Generator":
        st.header("Job Description Generator 📝")
        st.write("Enter the job title and any additional requirements to generate a high-quality job description.")

        job_title = st.text_input("Enter the job title:", placeholder="e.g., Senior Software Engineer")
        additional_requirements = st.text_area("Enter any additional requirements (optional):", 
                                               placeholder="e.g., TS clearance, 5+ years of experience in Python, knowledge of machine learning",
                                               key="job_description_requirements")

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

    elif tool_choice == "Prompt Generator":
        st.header("Prompt Generator 🧠")
        st.write("Generate and refine prompts for AI models.")

        if 'current_prompt' not in st.session_state:
            st.session_state.current_prompt = None
        if 'user_feedback' not in st.session_state:
            st.session_state.user_feedback = ""
        if 'improve_clicked' not in st.session_state:
            st.session_state.improve_clicked = False

        user_request = st.text_area("Enter your request for prompt generation:", 
                                    placeholder="e.g., Create a prompt for a coding assistant that specializes in Python and data analysis",
                                    key="prompt_generator_request")

        if st.button("Generate Initial Prompt"):
            st.session_state.current_prompt = None
            st.session_state.user_feedback = ""
            st.session_state.improve_clicked = False

            prompt_placeholder = st.empty()

            async def update_prompt():
                async for prompt in prompt_generator(user_request):
                    st.session_state.current_prompt = prompt
                    prompt_placeholder.text_area("Generated Prompt:", value=st.session_state.current_prompt, height=300, key="initial_prompt_output")
                    if not st.session_state.improve_clicked:
                        break

            asyncio.run(update_prompt())

        if st.session_state.current_prompt:
            st.session_state.user_feedback = st.text_area("Provide feedback to improve the prompt (or type 'end' to finish):", key="user_feedback_input")
            
            if st.button("Improve Prompt"):
                st.session_state.improve_clicked = True
                prompt_placeholder = st.empty()

                async def update_prompt():
                    async for prompt in prompt_generator(user_request):
                        st.session_state.current_prompt = prompt
                        prompt_placeholder.text_area("Improved Prompt:", value=st.session_state.current_prompt, height=400, key="improved_prompt_output")
                        if st.session_state.user_feedback.strip().lower() == 'end':
                            break

                asyncio.run(update_prompt())

    st.sidebar.title("About")
    st.sidebar.info(
        "This app provides various AI-powered tools for internal use. "
        "Currently, it includes a Job Description Generator and a Prompt Generator. "
        "Choose a tool from the sidebar to get started."
    )

if __name__ == "__main__":
    main()