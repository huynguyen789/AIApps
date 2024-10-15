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
import google.generativeai as genai
from docx import Document
import pypandoc
import tempfile
import anthropic
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from requests.exceptions import Timeout


# Add this function to set up the Gemini model
def setup_gemini_model():
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0,
        "max_output_tokens": 8192,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

async def search_and_summarize(query, model_choice, search_type, progress_callback=None):
    if progress_callback:
        await progress_callback("Searching for web content...", 0.1)
    
    # Serper API call
    url = "https://google.serper.dev/search"
    serper_api_key = st.secrets["SERPER_API_KEY"]
    
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        search_results = response.json()
    except requests.exceptions.RequestException as e:
        error_message = f"Error calling Serper API: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_message += f"\nResponse content: {e.response.text}"
        st.error(error_message)
        return 0, 0, ""
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from Serper API: {str(e)}\nResponse content: {response.text}")
        return 0, 0, ""

    if 'organic' not in search_results:
        st.error(f"Unexpected response format from Serper API. Response: {search_results}")
        return 0, 0, ""

    if progress_callback:
        await progress_callback("Processing links...", 0.2)
    combined_content = "Processed Links:\n"
    successful_website_count = 0
    successful_youtube_count = 0
    total_links = len(search_results['organic']) if search_type == "deep" else min(2, len(search_results['organic']))
    word_count = 0
    max_words = 50000

    for rank, result in enumerate(search_results['organic'][:total_links], 1):
        progress = 0.2 + (0.5 * rank / total_links)

        if word_count >= max_words:
            break

        if 'youtube.com' in result['link']:
            try:
                if progress_callback:
                    await progress_callback(f"Processing YouTube video {successful_youtube_count + 1} (Link {rank}/{total_links})", progress)
                video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', result['link'])
                if video_id_match:
                    video_id = video_id_match.group(1)
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = ' '.join([entry['text'] for entry in transcript])
                    transcript_words = transcript_text.split()
                    
                    if word_count + len(transcript_words) > max_words:
                        transcript_words = transcript_words[:max_words - word_count]
                        transcript_text = ' '.join(transcript_words)
                    
                    combined_content += f"{successful_youtube_count + 1}. [YouTube] {result['link']}\n"
                    combined_content += f"<youtube_video_{successful_youtube_count+1} rank={rank} title='{result['title']}' url='{result['link']}'>\n"
                    combined_content += transcript_text
                    combined_content += f"\n</youtube_video_{successful_youtube_count+1}>\n\n"
                    successful_youtube_count += 1
                    word_count += len(transcript_words)
            except Exception as e:
                pass
        
        else:
            try:
                if progress_callback:
                    await progress_callback(f"Processing website {successful_website_count + 1} (Link {rank}/{total_links})", progress)
                page_response = requests.get(result['link'], timeout=5)
                soup = BeautifulSoup(page_response.content, 'html.parser')
                cleaned_content = clean_content(soup)
                content_words = cleaned_content.split()
                
                if word_count + len(content_words) > max_words:
                    content_words = content_words[:max_words - word_count]
                    cleaned_content = ' '.join(content_words)
                
                combined_content += f"{successful_website_count + 1}. [Website] {result['link']}\n"
                combined_content += f"<website_{successful_website_count+1} rank={rank} url='{result['link']}'>\n"
                combined_content += cleaned_content
                combined_content += f"\n</website_{successful_website_count+1}>\n\n"
                successful_website_count += 1
                word_count += len(content_words)
            except Timeout:
                pass
            except Exception as e:
                pass

    combined_content += "\n\nProcessed Content:\n"

    if progress_callback:
        await progress_callback(f"Generating LLM response... (Used {successful_website_count} websites and {successful_youtube_count} YouTube videos)", 0.7)
    
    # Load the search prompt
    search_prompt = load_prompt('search_summarize.txt')
    
    # Format the prompt with the query and combined content
    formatted_prompt = search_prompt.format(
        query=query,
        combined_content=combined_content
    )
    
    # Use the selected model for generating the response
    if model_choice == "gemini":
        model = setup_gemini_model()
        response = model.generate_content(formatted_prompt, stream=True)
        response_container = st.empty()
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                response_container.markdown(full_response)
    elif model_choice == "claude":
        async with anthropic.AsyncClient(api_key=st.secrets["ANTHROPIC_API_KEY"]) as aclient:
            async with aclient.messages.stream(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4096,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ]
            ) as stream:
                response_container = st.empty()
                full_response = ""
                async for text in stream.text_stream:
                    full_response += text
                    response_container.markdown(full_response)
    elif model_choice == "gpt4":
        stream = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=4096,
            temperature=0,
            stream=True
        )
        response_container = st.empty()
        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                response_container.markdown(full_response)

    if progress_callback:
        await progress_callback("Search and summarize process completed", 1.0)

    return successful_website_count, successful_youtube_count, combined_content, full_response



#print out the raw content of this website:https://www.datacamp.com/tutorial/gpt4o-api-openai-tutorial
def print_raw_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        raw_content = soup.get_text()
        print(raw_content)
    except Exception as e:
        print(f"Error fetching content from {url}: {str(e)}")

# Add this line at the end of your script or where appropriate
print_raw_content("https://www.datacamp.com/tutorial/gpt4o-api-openai-tutorial")
