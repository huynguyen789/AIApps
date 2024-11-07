import anthropic
import base64
from pathlib import Path
import streamlit as st

def analyze_pdf_conversation(pdf_data_list, conversation_history, new_question):
    '''
    Input: List of PDF data (base64), conversation history, and new question
    Process: Maintains chat context while using prompt caching
    Output: Claude's response
    '''
    client = anthropic.Anthropic()
    
    # Create PDF document content list
    pdf_documents = [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": pdf_data
            },
            "cache_control": {"type": "ephemeral"}
        }
        for pdf_data in pdf_data_list
    ]
    
    response = client.beta.messages.create(
        model="claude-3-5-sonnet-20241022",
        betas=["pdfs-2024-09-25", "prompt-caching-2024-07-31"],
        max_tokens=3000,
        messages=[
            # First message with PDFs
            {
                "role": "user",
                "content": pdf_documents
            },
            # Previous conversation history
            *conversation_history,
            # New question
            {
                "role": "user",
                "content": [{"type": "text", "text": new_question}]
            }
        ]
    )
    
    return response.content[0].text

# Streamlit UI
st.title("Visual Document Chat Assistant")

# Add instruction message
st.markdown("""
📚 **Welcome!**
- This assistant can understand both text AND visual content (images, diagrams, charts)
- Perfect for analyzing documents containing visual information
""")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_data_list" not in st.session_state:
    st.session_state.pdf_data_list = []

# Add reset button in the sidebar
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.rerun()

# File uploader with session state
uploaded_files = st.file_uploader(
    "Upload your PDFs - Should be less than 31MB total", 
    type="pdf", 
    accept_multiple_files=True, 
    help="Limit 31MB per file • PDF",
    key="pdf_uploader"
)

if uploaded_files:
    # Clear existing PDFs if new ones are uploaded
    st.session_state.pdf_data_list = []
    
    for uploaded_file in uploaded_files:
        # Check file size (31MB limit)
        file_size = len(uploaded_file.read())
        uploaded_file.seek(0)  # Reset file pointer
        
        if file_size > 31 * 1024 * 1024:
            st.error(f"File '{uploaded_file.name}' exceeds 31MB limit. This file will be skipped.")
        else:
            # Add PDF data to session state list
            pdf_data = base64.b64encode(uploaded_file.read()).decode("utf-8")
            st.session_state.pdf_data_list.append(pdf_data)
    
    if st.session_state.pdf_data_list:
        st.success(f"Successfully loaded {len(st.session_state.pdf_data_list)} PDF(s)")

# Show chat interface if we have PDF data
if st.session_state.pdf_data_list:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your PDFs"):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get AI response
        try:
            response = analyze_pdf_conversation(
                st.session_state.pdf_data_list,  # Pass list of PDF data
                st.session_state.messages[:-1],
                prompt
            )
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Please upload one or more PDF files to start chatting!")

