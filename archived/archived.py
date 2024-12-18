

#Text to diagram
async def generate_mermaid_diagram(description):
    prompt = """
    You are an expert in creating Mermaid diagrams. Based on the user's description, generate a Mermaid diagram code.
    Make sure the code is valid and follows Mermaid syntax. Return only the Mermaid code, without any additional text or explanations, tags, or code block markers.
    If the chart getting weirdly too long, make sure to design it so it fit nicely in user monitor(not super long that they have to scroll too much)
    

    
    Good output:
        graph TD
            A[Start] --> B[Process 1]
            B --> C[Process 2]
            C --> D[End]
            
            
    DO NOT INCLUDE THE "```mermaid"
    """
    
    user_input = f"Create a Mermaid diagram for: {description}"
    
    response = await anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
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
#########################################################
  

#Writing tools
async def process_text(user_input, task):
    prompts = {
        "grammar": "Correct the grammar in the following text...",
        "rewrite": "Rewrite the following text professionally...",
        "summarize": "Summarize the following text concisely...",
        "explain": "Explain the following text in two parts..."
    }
   
    user_input = f'"{user_input}"'
    
    async for content in generate_response("gpt4", user_input, system_prompt=prompts[task]):
        yield content

    # return processed_content
#########################################################


  

elif tool_choice == "Diagram Creation Assistant":
    st.header("Diagram Creation Assistant 🎨")
    st.write("Transform your ideas into visual diagrams with ease!")
    st.write("Here are some example descriptions you can use:")
    st.code("""
    1. A flowchart of making coffee
    2. A flowchart of an MLOps pipeline
    3. A flowchart of a user registration process
    """, language="markdown")

    # Initialize session state variables
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

    if st.session_state.mermaid_code:
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
                    mime="image/png",
                    key="png_download"
                )
            else:
                st.warning("Unable to generate a downloadable PNG. You can still use the Mermaid code above.")

    if st.button("Reset"):
        st.session_state.diagram_description = ""
        st.session_state.mermaid_code = ""
        st.rerun()

   
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

        # File upload option for user input
        input_file = st.file_uploader("Upload your text (Word or Text file)", type=['docx', 'txt'])
        if input_file:
            st.session_state.user_input = read_file(input_file)
        
        user_input = st.text_area("Or enter your text here:", 
                                  value=st.session_state.user_input,
                                  height=200, 
                                  key="user_input")
        
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
