import streamlit as st
import os
import asyncio
import json
import re
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import streamlit_mermaid as stmd
import base64
import requests

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



#Job description generator:
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
#End Job description generator


#Writing tools
async def process_text(user_input, task):
   
    prompts = {
        "grammar": "Correct the grammar in the following text, return the correct content, output in a format that's easy for the user to copy. Also, show what was corrected:",
        "rewrite": "Rewrite the following text professionally and concisely. Maintain the core message. Give few answers so user can select instead of only 1 option",
        "summarize": "Summarize the following text concisely, capturing the main points:",
        "explain": """Explain the following text in two parts in a super easy to way to understand and concise:
        1. Explain it in a super simple way like the user is a 12 years old, with example.
        2. Then, explain it in regular way.
        
        Make sure both explanations are clear and easy to understand."""
    }
   
    user_input = f'"{user_input}"'
    
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
            yield chunk.choices[0].delta.content

    # return processed_content
#End Writing tools

#BD Response Assistant
async def get_feedback(document, requirements):
    user_content = f"""
    <system_prompt>
    You are an expert in business development proposals for government with 30 years track record exceptional experience and of 95% winning rate proposal."
    </system_prompt>
    
    Requirements Document:\n{requirements}\n\nUser's Draft:\n{document}\n\n
    
    Provide detailed feedback on the user's draft generally and on the requirements provided. 
    Be critical, point out things that user dont see. The goal is a world-class winning proposal. 
    MAKE SURE YOU DOING AN EXCEPTIONAL JOB, I WILL GIVE YOU $100,000 BONUS THIS YEAR, IF NOT A CAT WILL DIE."""

    response = await openai_client.chat.completions.create(
        model="o1-preview",
        messages=[
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content
#End BD Response Assistant



#Text to diagram
async def generate_mermaid_diagram(description):
    prompt = """
    You are an expert in creating Mermaid diagrams. Based on the user's description, generate a Mermaid diagram code.
    Make sure the code is valid and follows Mermaid syntax. Return only the Mermaid code, without any additional text or explanations, tags, or code block markers.
    Try to design the code to fit nicely in horizontal monitor for user.
    

    
    Good output:
        graph TD
            A[Start] --> B[Process 1]
            B --> C[Process 2]
            C --> D[End]
            
            
    DO NOT INCLUDE THE "```mermaid"
    """
    
    user_input = f"Create a Mermaid diagram for: {description}"
    
    response = await anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=3000,
        temperature=0,
        system=prompt,
        messages=[
            {
                "role": "user",
                "content": user_input
            }
        ]
    )
    return response.content[0].text

def mermaid_to_svg(mermaid_code):
    response = requests.post(
        'https://mermaid.ink/svg',
        json={'mermaid': mermaid_code}
    )
    if response.status_code == 200 and response.content.startswith(b'<svg'):
        return response.content
    else:
        return None
    
def generate_mermaid_chart(mermaid_code, format='png'):
    # Mermaid Live Editor API endpoint
    url = 'https://mermaid.ink/img/'

    # Encode the Mermaid code
    encoded_code = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')

    # Construct the full URL
    full_url = f"{url}{encoded_code}"

    # Add format parameter if not PNG
    if format.lower() != 'png':
        full_url += f"?type={format}"

    # Send GET request to the API
    response = requests.get(full_url)

    # Check if the request was successful
    if response.status_code == 200:
        return response.content
    else:
        return None    
    
    

async def streamlit_main():
    st.set_page_config(page_title="AI Assistant Tools", page_icon="🛠️", layout="wide")

    st.title("AI Assistant Tools 🛠️")
    st.markdown(f"""
        This app provides various AI-powered tools for internal use. 
        \nCurrently, it includes a Job Description Generator, a Prompt Generator, a Writing Assistant, and a BD Response Assistant. 
        \nChoose a tool from the sidebar to get started.
        """
    )
    
    tool_choice = st.sidebar.radio("Choose a tool:", ("Job Description Generator", "Prompt Generator", "Writing Assistant", "BD Response Assistant", "Text to Diagram Converter"))

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
                
        feedback = st.text_area("Provide feedback to improve the job description:", placeholder="example: tailor for CBM+ projects, 7 years of experience instead of 5, etc...")
        
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
   
    elif tool_choice == "Writing Assistant":
        st.header("Writing Assistant ✍️")
        st.write("Welcome to the Writing Assistant! Here you can improve your writing with AI-powered tools. Choose from the following options:")
        st.write("1. Professional Rewrite: Enhance the professionalism of your text.")
        st.write("2. Correct Grammar: Fix grammatical errors in your text.")
        st.write("3. Summarize: Get a concise summary of your text.")
        st.write("4. Explain: Simplify and explain your text.")

        if 'user_input' not in st.session_state:
            st.session_state.user_input = ""
        if 'task' not in st.session_state:
            st.session_state.task = "Professional Rewrite"
        if 'processed_text' not in st.session_state:
            st.session_state.processed_text = ""

        user_input = st.text_area("Enter your text (e.g., 'I ain't got no time for this.'):", height=200, key="user_input")
        
        task = st.selectbox("Choose a task:", ["Professional Rewrite", "Correct Grammar", "Summarize", "Explain"], key="task")
            
        if st.button("Process Text"):
            if user_input:
                task_map = {
                    "Correct Grammar": "grammar",
                    "Professional Rewrite": "rewrite",
                    "Summarize": "summarize",
                    "Explain": "explain"
                }
        
                with st.spinner(f"Processing text ({task.lower()})..."):
                    processed_text_placeholder = st.empty()
                    full_content = ""
                    async for content in process_text(user_input, task_map[task]):
                        full_content += content
                        processed_text_placeholder.markdown(full_content)
                    st.session_state.processed_text = full_content

        # if st.session_state.processed_text:
            # st.markdown(f"### {task} Result:")
            # st.write(st.session_state.processed_text)

    elif tool_choice == "BD Response Assistant":
        st.header("BD Response Assistant 📄")
        
        # Initialize session state variables for BD Response Assistant
        if "bd_document" not in st.session_state:
            st.session_state.bd_document = ""
        if "bd_requirements" not in st.session_state:
            st.session_state.bd_requirements = ""

        st.write("Paste your BD response draft and the requirements document to receive AI-generated feedback.")

        # Use session state for document and requirements
        st.session_state.bd_document = st.text_area("Paste your BD Response Draft here:", 
                                                    value=st.session_state.bd_document, 
                                                    height=300)
        st.session_state.bd_requirements = st.text_area("Paste the Requirements Document here:", 
                                                        value=st.session_state.bd_requirements, 
                                                        height=300)

        if st.button("Get Feedback"):
            if st.session_state.bd_document.strip() == "" or st.session_state.bd_requirements.strip() == "":
                st.warning("Please paste both the BD response draft and the requirements document.")
            else:
                with st.spinner("Generating feedback..."):
                    feedback = await get_feedback(st.session_state.bd_document, st.session_state.bd_requirements)
                    st.markdown(feedback)

    elif tool_choice == "Text to Diagram Converter":
        st.header("Text to Diagram Converter 🎨")
        st.write("Transform your ideas into visual diagrams with ease!")
        st.write("Here are some example descriptions you can use:")
        st.code("""
        1. A flowchart of making coffee
        2. A flowchart of an MLOps pipeline
        3. A flowchart of a user registration process
        """, language="markdown")

        # Initialize session state for diagram description and Mermaid code
        if "diagram_description" not in st.session_state:
            st.session_state.diagram_description = ""
        if "mermaid_code" not in st.session_state:
            st.session_state.mermaid_code = ""

        # Use session state for the text area
        st.session_state.diagram_description = st.text_area(
            "Describe the diagram you want to create:", 
            value=st.session_state.diagram_description,
            placeholder="e.g., A flowchart showing the steps to plan a vacation, a flowchart of a MLOps system"
        )

        if st.button("Create Diagram"):
            if st.session_state.diagram_description:
                with st.spinner("Creating your diagram..."):
                    st.session_state.mermaid_code = await generate_mermaid_diagram(st.session_state.diagram_description)
                    st.subheader("Your Generated Diagram:")
                    try:
                        mermaid = stmd.st_mermaid(st.session_state.mermaid_code, height=800)
                    except Exception as e:
                        st.error(f"Oops! There was an error creating your diagram: {str(e)}")
                        st.text("Technical details (for troubleshooting):")
                        st.code(st.session_state.mermaid_code, language="mermaid")
                    else:
                        st.subheader("Diagram Code:")
                        st.code(st.session_state.mermaid_code, language="mermaid")
                        st.info("You can copy this code and edit the diagram in [Mermaid Live](https://mermaid.live/).")
                        
                        # Generate PNG image
                        png_image = generate_mermaid_chart(st.session_state.mermaid_code, format='png')
                        if png_image:
                            st.download_button(
                                label="Download Diagram as PNG",
                                data=png_image,
                                file_name="mermaid_diagram.png",
                                mime="image/png"
                            )
                        else:
                            st.warning("Unable to generate a downloadable PNG. You can still use the Mermaid code above.")

                        # Generate SVG image
                        svg_image = generate_mermaid_chart(st.session_state.mermaid_code, format='svg')
                        if svg_image:
                            st.download_button(
                                label="Download Diagram as SVG",
                                data=svg_image,
                                file_name="mermaid_diagram.svg",
                                mime="image/svg+xml"
                            )
                        else:
                            st.warning("Unable to generate a downloadable SVG. You can still use the Mermaid code above.")

            else:
                st.warning("Please enter a description for your diagram.")


if __name__ == "__main__":
    asyncio.run(streamlit_main())