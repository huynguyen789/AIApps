import anthropic
import base64
from pathlib import Path
import streamlit as st

def analyze_pdf_conversation(pdf_data, conversation_history, new_question):
    '''
    Input: PDF data (base64), conversation history, and new question
    Process: Maintains chat context while using prompt caching
    Output: Claude's response
    '''
    client = anthropic.Anthropic()
    
    response = client.beta.messages.create(
        model="claude-3-5-sonnet-20241022",
        betas=["pdfs-2024-09-25", "prompt-caching-2024-07-31"],
        max_tokens=2000,
        messages=[
            # First message with PDF
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data
                        },
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
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
st.title("Chat with your PDF")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None

# Add reset button in the sidebar
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.rerun()

# File uploader with session state
uploaded_file = st.file_uploader("Upload your PDF", type="pdf", key="pdf_uploader")

if uploaded_file is not None:
    # Check file size (31MB = 31 * 1024 * 1024 bytes)
    file_size = len(uploaded_file.read())
    uploaded_file.seek(0)  # Reset file pointer after reading
    
    if file_size > 31 * 1024 * 1024:
        st.error("File size exceeds 31MB limit. Please upload a smaller PDF.")
    else:
        # Store PDF data in session state
        st.session_state.pdf_data = base64.b64encode(uploaded_file.read()).decode("utf-8")

# Show chat interface if we have PDF data
if st.session_state.pdf_data:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF"):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get AI response
        try:
            response = analyze_pdf_conversation(
                st.session_state.pdf_data,  # Use stored PDF data
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
    st.info("Please upload a PDF file to start chatting!")

