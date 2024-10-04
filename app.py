import streamlit as st
import os
import asyncio
import json
import re
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# Initialize clients
openai_client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = AsyncAnthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

def load_prompt(filename):
    with open(os.path.join('./prompts', filename), 'r') as file:
        return file.read()

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
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

def remove_analysis(prompt):
    return re.sub(r'<analysis>.*?</analysis>', '', prompt, flags=re.DOTALL)

async def generate_job_description(job_title, additional_requirements):
    main_prompt = load_prompt('job_description.txt')
    
    user_input = f"Job title: {job_title}. {additional_requirements}"
    
    response = await openai_client.chat.completions.create(
        model="gpt-4o",  # Changed from "gpt-4o" to "gpt-4"
        messages=[
            {"role": "system", "content": main_prompt},
            {"role": "user", "content": user_input}
        ],
        stream=True,
    )

    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
    
async def improve_job_description(original_jd, feedback, job_title, additional_requirements):
    improve_prompt = load_prompt('improve_job_description.txt')
    improve_input = f"""
Original Job Title: {job_title}
Additional Requirements: {additional_requirements}

Original Job Description:
{original_jd}

User Feedback:
{feedback}
"""
    response = await openai_client.chat.completions.create(
        model="gpt-4o",  # Changed from "gpt-4o" to "gpt-4"
        messages=[
            {"role": "system", "content": improve_prompt},
            {"role": "user", "content": improve_input}
        ],
        stream=True,
    )
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
async def prompt_generator(user_request):
    optimizer_model = "claude-3-sonnet-20240229"  # or "gpt-4-0125-preview"
    
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = None
    if 'user_feedback' not in st.session_state:
        st.session_state.user_feedback = ""
    
    # Generate initial prompt if it doesn't exist
    if st.session_state.current_prompt is None:
        initial_improvement_prompt = load_prompt('create_adv_prompt.txt')
        initial_improvement_prompt = initial_improvement_prompt.format(
            user_request=user_request,
            user_feedback="No user feedback yet",
            current_prompt="No current prompt yet."
        )
        
        with st.spinner("Generating initial prompt..."):
            initial_prompt = await get_model_answer(initial_improvement_prompt, "", optimizer_model)
        
        st.session_state.current_prompt = remove_analysis(initial_prompt)

    st.subheader("Current Prompt:")
    st.text_area("Current Prompt", value=st.session_state.current_prompt, height=300, key="current_prompt_display")
    
    # Capture user feedback
    user_feedback = st.text_area("Provide feedback for improving the prompt:", value=st.session_state.user_feedback, key="user_feedback_input")
    
    col1, col2 = st.columns(2)
    improve_button = col1.button("Improve Prompt")
    reset_button = col2.button("Reset")
    
    if improve_button:
        st.session_state.user_feedback = user_feedback
        improvement_prompt = load_prompt('create_adv_prompt.txt')
        improvement_prompt = improvement_prompt.format(
            user_request=user_request,
            user_feedback=f"User feedback: {st.session_state.user_feedback}",
            current_prompt=st.session_state.current_prompt
        )
        
        with st.spinner("Generating improved prompt..."):
            improved_prompt = await get_model_answer(improvement_prompt, "", optimizer_model)
            st.session_state.current_prompt = remove_analysis(improved_prompt)
        
        st.session_state.user_feedback = ""
        st.rerun()
    
    if reset_button:
        st.session_state.current_prompt = None
        st.session_state.user_feedback = ""
        st.rerun()
    
    return st.session_state.current_prompt


#Writing tools
async def process_text(user_input, task):
    prompts = {
        "grammar": "Correct the grammar in the following text, return the correct content, output in a format that's easy for the user to copy. Also, show what was corrected:",
        "rewrite": "Rewrite the following text professionally and concisely. Maintain the core message while improving clarity and brevity:",
        "summarize": "Summarize the following text concisely, capturing the main points:",
        "explain": """Explain the following text in two parts:
        1. Explain it as if you're talking to a 10-year-old child.
        2. Then, explain it for an adult audience.
        
        Make sure both explanations are clear and easy to understand."""
    }

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompts[task]},
            {"role": "user", "content": user_input}
        ],
        stream=True,
    )
    processed_content = ""

    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            processed_content += chunk.choices[0].delta.content

    return processed_content
#End Writing tools



async def streamlit_main():
    st.set_page_config(page_title="AI Assistant Tools", page_icon="🛠️", layout="wide")

    st.title("AI Assistant Tools 🛠️")
    st.markdown("""
    Streamline your AI interactions with our advanced prompt engineering automation. Simply input your desired prompt type, and let our app craft highly effective prompts using sophisticated techniques. From job descriptions to custom AI instructions, we've got you covered.
    """)
    st.sidebar.title("About")
    st.sidebar.info(
        "This app provides various AI-powered tools for internal use. "
        "Currently, it includes a Job Description Generator and a Prompt Generator. "
        "Choose a tool from the sidebar to get started."
    )
    
    
    tool_choice = st.sidebar.radio("Choose a tool:", ("Job Description Generator", "Prompt Generator", "Writing Assistant"))

    if tool_choice == "Prompt Generator":
        st.header("Prompt Generator 🧠")
        st.write("Generate and refine prompts for AI models.")

        user_request = st.text_input("Enter your request for prompt generation:")
        if user_request:
            await prompt_generator(user_request)
            


    elif tool_choice == "Job Description Generator":
        st.header("Job Description Generator 📝")
        st.write("Enter the job title and any additional requirements to generate a high-quality job description.")

        if 'job_description' not in st.session_state:
            st.session_state.job_description = ""
        if 'job_title' not in st.session_state:
            st.session_state.job_title = ""
        if 'additional_requirements' not in st.session_state:
            st.session_state.additional_requirements = ""

        job_title = st.text_input("Enter the job title:", 
                                  value=st.session_state.job_title, 
                                  placeholder="e.g., Senior Software Engineer",
                                  key="job_title_input")  # Added unique key
        additional_requirements = st.text_area("Enter any additional requirements (optional):", 
                                               value=st.session_state.additional_requirements,
                                               placeholder="You can paste the PWS requirements here OR manually type: TS clearance, 5+ years of experience in Python, knowledge of machine learning, etc.",
                                               key="job_description_requirements")
        st.info("Note: If you want the tool to strictly follow the requirements, include 'PWS' in the additional requirements box.")



        if st.button("Generate Job Description"):
            if job_title:
                with st.spinner("Generating job description..."):
                    job_description_placeholder = st.empty()
                    full_content = ""
                    async for content in generate_job_description(job_title, additional_requirements):
                        full_content += content
                        job_description_placeholder.markdown(full_content)
                    st.session_state.job_description = full_content
                    st.session_state.job_title = job_title
                    st.session_state.additional_requirements = additional_requirements
            else:
                st.warning("Please enter a job title.")
        
        # Display the current job description
        # if st.session_state.job_description:
        #     st.markdown("### Current Job Description")
        #     st.markdown(st.session_state.job_description)
                
        feedback = st.text_area("Provide feedback to improve the job description:", placeholder="What would you like to change or improve?")
        
        if st.button("Improve Job Description"):
            if feedback and st.session_state.job_description:
                with st.spinner("Improving job description..."):
                    improved_jd_placeholder = st.empty()
                    improved_content = ""
                    async for content in improve_job_description(st.session_state.job_description, feedback, st.session_state.job_title, st.session_state.additional_requirements):
                        improved_content += content
                        improved_jd_placeholder.markdown(improved_content)
                    st.session_state.job_description = improved_content
            elif not st.session_state.job_description:
                st.warning("Please generate a job description first.")
            else:
                st.warning("Please provide feedback to improve the job description.")

        if tool_choice == "Writing Assistant":
            st.header("Writing Assistant ✍️")
            st.write("Improve your writing with AI-powered grammar correction.")

            user_input = st.text_area("Enter your text for grammar correction:", height=200)
            
            if st.button("Correct Grammar"):
                if user_input:
                    with st.spinner("Correcting grammar..."):
                        corrected_text = await grammar_corrector(user_input)
                        st.markdown("### Corrected Text:")
                        st.write(corrected_text)
                else:
                    st.warning("Please enter some text to correct.")


    elif tool_choice == "Writing Assistant":
        st.header("Writing Assistant ✍️")
        st.write("Improve your writing with AI-powered tools.")

        user_input = st.text_area("Enter your text:", height=200)
        
        col1, col2 = st.columns(2)
        task = col1.selectbox("Choose a task:", ["Correct Grammar", "Professional Rewrite", "Summarize", "Explain"])
        
        if col2.button("Process Text"):
            if user_input:
                task_map = {
                    "Correct Grammar": "grammar",
                    "Professional Rewrite": "rewrite",
                    "Summarize": "summarize",
                    "Explain": "explain"
                }
                with st.spinner(f"Processing text ({task.lower()})..."):
                    processed_text = await process_text(user_input, task_map[task])
                    st.markdown(f"### {task} Result:")
                    st.write(processed_text)
            else:
                st.warning("Please enter some text to process.")
                    


if __name__ == "__main__":
    asyncio.run(streamlit_main())