import anthropic
import base64
from pathlib import Path

def analyze_pdf(pdf_path, question):
    """
    Input: Path to PDF file and question to ask about the PDF
    Process: Reads PDF, then sends to Claude API with caching enabled
    Output: Claude's response about the PDF
    """
    pdf_data = base64.b64encode(Path(pdf_path).read_bytes()).decode("utf-8")
    client = anthropic.Anthropic()
    
    # Make request with prompt caching enabled
    message = client.beta.messages.create(
        model="claude-3-5-sonnet-20241022",
        betas=["pdfs-2024-09-25", "prompt-caching-2024-07-31"],
        max_tokens=2000,
        messages=[
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
                        "cache_control": {"type": "ephemeral"}  # Enable caching for the PDF
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
    )
    
    return message.content[0].text

# Example usage of single query
pdf_path = "/Users/huyknguyen/Desktop/redhorse/code_projects/ai_apps/docs/redhorse_docs/2024:10:2025-Workweek-Calendar.pdf"
question = "Please summarize the key dates and information from this calendar."

try:
    response = analyze_pdf(pdf_path, question)
    print(response)
except Exception as e:
    print(f"Error: {e}")
