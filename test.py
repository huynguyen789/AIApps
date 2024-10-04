import streamlit as st
import streamlit_mermaid as stmd

st.title("Mermaid Diagram Example")

mermaid_code = """
graph TD
    A[Start] --> B[Process 1]
    B --> C[Process 2]
    C --> D[End]
"""

st.write("Here's a simple Mermaid diagram:")
mermaid = stmd.st_mermaid(mermaid_code)

# You can also write the result if needed
st.write(mermaid)