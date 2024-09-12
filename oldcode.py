import streamlit as st
import os
import yaml
from openai import OpenAI
import time



# Set page config at the very beginning
st.set_page_config(page_title="Character Chat System", page_icon="🧠", layout="wide")


# Initialize OpenAI client
model_name = "gpt-4o"
# openai_api_key = os.environ.get("OPENAI_API_KEY")

openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
client = OpenAI(api_key=openai_api_key)


def load_prompt(filename):
    with open(os.path.join('./prompts', filename), 'r') as file:
        return file.read()
    
def list_existing_prompts():
    characters = []
    for filename in os.listdir('./few_shot_prompts'):
        if filename.endswith('.yaml'):
            characters.append(filename)
    return characters

def check_and_summarize_memory(messages):
    # Calculate total character count
    total_chars = sum(len(message['content']) for message in messages)
    
    # If character count is approaching 100,000 (check at 80,000 to leave room for response)
    if total_chars > 80000:
        
        #print full messages for debugging
        print("Full messages:")
        for msg in messages:
            print(f"Message: {msg}")
            print("-" * 50)
        
        # join all messages into a single string
        conversation_messages = "\n".join([msg['content'] for msg in messages])

        # Prepare the summarization prompt
        summary_prompt = """Summarize the content below. 
        - Keep the 'persona info:' as close as possible to the original. If it is too long you can summarize it.
        - Keep the the 'Instructions:' exactly as it is. 
        - Summarize the conversation, highlight the important details and info.
        - Do not include the json tags or any other text.
        Output example:
            [{"role": "system", "content": "persona info: ....\nInstructions:...."},
            {"role": "assistant", "content": "Summarized conversation"},
             {"role": "user", "content": "(put user newest message here)"}
            ]
        """

        # Prepare messages for OpenAI API
        summarization_messages = [
            {"role": "user", "content": summary_prompt + "\n\n" + str(conversation_messages)}
        ]

        # Call OpenAI API for summarization
        print(f"Conversation getting too long, summarizing...")
        response = client.chat.completions.create(
            model=model_name,
            messages=summarization_messages
        )
        
        
        # Extract the summary from the response
        summary = response.choices[0].message.content
        print(f"Summary: {summary}")
        
        # Parse the summary back into a list of message dictionaries
        try:
            new_messages = eval(summary)
            if not isinstance(new_messages, list) or not all(isinstance(m, dict) for m in new_messages):
                raise ValueError("Invalid summary format")
        except:
            # Fallback if parsing fails
            new_messages = [{"role": "system", "content": summary}]

        print(f"New messages: {new_messages}")
  
        return new_messages, True

    return messages, False

def extract_character_name(character_data):
    if isinstance(character_data, dict) and 'content' in character_data:
        content = character_data['content']
        
        # # Remove ```yaml if present at the start
        # content = content.lstrip('`').lstrip()
        # if content.startswith('yaml'):
        #     content = content[4:].lstrip()
        
        # Look for "name: " or "Name: " followed by the name
        name_start = content.find("name:")
        if name_start == -1:
            name_start = content.find("Name:")
        
        if name_start != -1:
            # Move to the start of the actual name
            name_start += 5  # length of "name:"
            # Find the end of the name (next newline or end of string)
            name_end = content.find("\n", name_start)
            if name_end == -1:
                name_end = len(content)
            
            return content[name_start:name_end].strip()

    # If we couldn't extract from content, use the filename
    if isinstance(character_data, dict) and 'filename' in character_data:
        # Extract name from filename
        filename = character_data['filename']
        if filename.startswith("withExamples_"):
            name_end = filename.find("_", len("withExamples_"))
            if name_end != -1:
                return filename[len("withExamples_"):name_end].replace('_', ' ')

    # If all else fails, return a generic name
    return "Unnamed Character"

def save_character_prompt(character_prompt, user_input):
    os.makedirs('./few_shot_prompts', exist_ok=True)
    filename = f"./few_shot_prompts/{user_input.replace(' ', '_')}.yaml"
    with open(filename, 'w') as file:
        yaml.dump({"character_prompt": character_prompt}, file)

def clean_character_name(filename):
    # Remove file extension
    name = os.path.splitext(filename)[0]
    # Remove 'withExamples_' prefix if present
    if name.startswith('withExamples_'):
        name = name[13:]
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    # Capitalize each word
    name = name.title()
    return name

def get_character_selection(existing_prompts, single=True):
    # Create a dictionary mapping cleaned names to original filenames
    name_to_file = {clean_character_name(file): file for file in existing_prompts}
    
    # Sort the cleaned names
    sorted_names = sorted(name_to_file.keys())
    
    if single:
        options = [""] + sorted_names
        selected_name = st.selectbox("Select from the dropdown or type your character's name to search", options, index=0)
        return name_to_file[selected_name] if selected_name else ""
    else:
        return [name_to_file[name] for name in st.multiselect("Choose characters", sorted_names)]

def load_character_prompt(filename):
    file_path = f"./few_shot_prompts/{filename}"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
                if isinstance(data, dict) and 'character_prompt' in data:
                    return data['character_prompt']
                else:
                    return file.read()
            except yaml.YAMLError as e:
                print(f"Error parsing YAML: {e}")
                return file.read()
    return None


class StreamingCharacterChatbot:
    def __init__(self, character_prompt, color='yellow'):
        self.character_prompt = character_prompt
        self.color = color
        system_prompt_template = load_prompt('single_character_system_prompt.txt')
        self.system_message = {"role": "system", "content": system_prompt_template.format(character_prompt=self.character_prompt)}
        self.messages = [self.system_message]

    def get_streaming_response(self):
        response = client.chat.completions.create(
            model=model_name,
            messages=self.messages,
            stream=True,
        )
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
        self.messages.append({"role": "assistant", "content": full_response})

    def start_conversation(self):
        self.messages.append({"role": "user", "content": "Please introduce yourself in 1-2 sentences."})
        return self.get_streaming_response()

    def get_response(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        self.messages, summarized = check_and_summarize_memory(self.messages)
        return self.get_streaming_response(), summarized


def select_characters():
    st.subheader("Select Character(s)")
    existing_prompts = list_existing_prompts()

    if st.session_state.chat_mode == "Single Character":
        selected_character = get_character_selection(existing_prompts, single=True)
        if st.button("Start Chat", key="start_single"):
            if selected_character:  # Only proceed if a character is selected
                st.session_state.selected_characters = [load_character_prompt(selected_character)]
                st.session_state.app_mode = "chat"
                initialize_chat()
                st.rerun()  # Add this line to force a rerun
            else:
                st.warning("Please select a character before starting the chat.")
    else:
        selected_characters = get_character_selection(existing_prompts, single=False)
        col1, col2 = st.columns(2)
        with col2:
            if st.button("Start Chat", key="start_multi"):
                if selected_characters:  # Only proceed if characters are selected
                    st.session_state.selected_characters = [load_character_prompt(char) for char in selected_characters]
                    st.session_state.app_mode = "chat"
                    initialize_chat()
                    st.rerun()  # Add this line to force a rerun
                else:
                    st.warning("Please select at least one character before starting the chat.")

class StreamingMultiCharacterChatbot:
    def __init__(self, characters):
        self.characters = characters
        self.character_names = [extract_character_name({'content': character}) for character in characters]
        self.character_colors = ['yellow', 'blue', 'green', 'magenta', 'cyan']
        self.chatbots = [StreamingCharacterChatbot(character, color) for character, color in zip(characters, self.character_colors)]
        
        self.moderator_messages = [
            {"role": "system", "content": "You are a world-class moderator tasked with synthesizing character opinions. "},
        ]

    def start_conversation(self):
        introductions = []
        for i, chatbot in enumerate(self.chatbots):
            with st.chat_message("assistant"):
                st.markdown(f"**{self.character_names[i]}**:")
                intro_generator = chatbot.start_conversation()
                intro = st.write_stream(intro_generator)
                introductions.append(intro)
            st.session_state.messages.append({"role": "assistant", "content": f"**{self.character_names[i]}**: {intro}"})
        return introductions

    def get_responses(self, user_input):
        responses = []
        for i, chatbot in enumerate(self.chatbots):
            with st.chat_message("assistant"):
                st.markdown(f"**{self.character_names[i]}**:")
                response_generator, summarized = chatbot.get_response(user_input)
                response = st.write_stream(response_generator)
                responses.append(response)
            st.session_state.messages.append({"role": "assistant", "content": f"**{self.character_names[i]}**: {response}"})
        return responses
    
    def get_discussion_responses(self, initial_responses):
        discussion_prompt = load_prompt('discussion_prompt.txt')
        for i, response in enumerate(initial_responses):
            discussion_prompt += f"\n\n{self.character_names[i]}: {response}"
        
        for i, chatbot in enumerate(self.chatbots):
            with st.chat_message("assistant"):
                st.markdown(f"**{self.character_names[i]}** (Discussion):")
                response_generator, summarized = chatbot.get_response(discussion_prompt)
                response = st.write_stream(response_generator)
            st.session_state.messages.append({"role": "assistant", "content": f"**{self.character_names[i]}** (Discussion): {response}"})


    def get_moderator_response(self, character_responses):
        synthesis_prompt = "Provide a short and concise summary of the characters' views, highlighting insights, areas of agreement and disagreement:\n\n"
        for i, response in enumerate(character_responses):
            synthesis_prompt += f"{self.character_names[i]}: {response}\n\n"
        
        self.moderator_messages.append({"role": "user", "content": synthesis_prompt})
        moderator_response = client.chat.completions.create(
            model=model_name,
            messages=self.moderator_messages,
            stream=True,
        )
        full_response = ""
        for chunk in moderator_response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
        self.moderator_messages.append({"role": "assistant", "content": full_response})
        return full_response


def initialize_session_state():
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = None
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = None
    if "previous_chat_mode" not in st.session_state:
        st.session_state.previous_chat_mode = None
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_characters" not in st.session_state:
        st.session_state.selected_characters = []
    if "use_moderator" not in st.session_state:
        st.session_state.use_moderator = True
    if "use_discussion" not in st.session_state:
        st.session_state.use_discussion = False
 
def create_character_prompt_stream(user_input):
    create_character_prompt_template = load_prompt('create_character_prompt_template.txt')
    
    formatted_prompt = create_character_prompt_template.format(user_input=user_input)
    messages = [{"role": "user", "content": formatted_prompt}]
    
    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
    )
    
    return stream


def initialize_chat():
    if st.session_state.chat_mode == "Single Character":
        character_prompt = st.session_state.selected_characters[0]
        st.session_state.chatbot = StreamingCharacterChatbot(character_prompt)
    else:
        st.session_state.chatbot = StreamingMultiCharacterChatbot(st.session_state.selected_characters)
    
    st.session_state.messages = []

def reset_chat():
    st.session_state.app_mode = "select_characters"
    st.session_state.selected_characters = []
    st.session_state.chatbot = None
    st.session_state.messages = []

def restart_conversation():
    if st.session_state.chatbot:
        st.session_state.messages = []
        if st.session_state.chat_mode == "Single Character":
            st.session_state.chatbot = StreamingCharacterChatbot(st.session_state.selected_characters[0])
        else:
            st.session_state.chatbot = StreamingMultiCharacterChatbot(st.session_state.selected_characters)
        st.rerun()

def create_new_character():
    st.subheader("Create New Expert")
    
    name = st.text_input("Enter expert type (e.g., 'Russian Military Expert'):")
    goal = st.text_input("Enter the goal of this conversation: (e.g., 'understand Russian perspective'):")
    
    example_file = st.file_uploader("Upload example file (optional)", type=['txt'])
    
    col1, col2 = st.columns(2)
    with col1:
        create_and_chat_button = st.button("Create Expert and Start Chat")
        
    with col2:
        create_button = st.button("Create Expert Only")
        
    
    if create_button or create_and_chat_button:
        if name and goal:
            user_input = f"Expert name:{name}.\nGoal of the conversation with this expert:{goal}"
            with st.spinner("Creating expert... This may take a moment."):
                try:
                    stream = create_character_prompt_stream(user_input)
                    
                    character_prompt_placeholder = st.empty()
                    character_prompt = ""
                    
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            character_prompt += content
                            character_prompt_placeholder.text(character_prompt)
                    
                    if example_file is not None:
                        example_content = example_file.getvalue().decode()
                        character_prompt += f"\n\nHere are a few examples of the character conversation or extra data:\n{example_content}"
                        file_prefix = "withExamples_"
                    else:
                        file_prefix = ""
                    
                    # Check and truncate name and goal if they're too long
                    max_length = 30  # You can adjust this value as needed
                    truncated_name = name[:max_length].replace(' ', '_')
                    truncated_goal = goal[:max_length].replace(' ', '_')
                    
                    filename = f"{file_prefix}{truncated_name}_{truncated_goal}.yaml"
                    save_character_prompt(character_prompt, filename)
                    
                    if create_button:
                        st.success(f"Expert '{name}' created successfully!\nClick 'Start Chatting' on the top left to begin a conversation with this character.")
                    elif create_and_chat_button:
                        st.success(f"Expert '{name}' created successfully! Starting chat...")
                        st.session_state.app_mode = "chat"
                        st.session_state.chat_mode = "Single Character"
                        st.session_state.selected_characters = [character_prompt]
                        initialize_chat()
                        st.rerun()
                except Exception as e:
                    st.error(f"An error occurred while creating the character: {str(e)}")
        else:
            st.warning("Please enter both the character type and goal.")

def chat_interface():
    st.subheader("Expert Chat")
    
    # Ensure chatbot is initialized
    if st.session_state.chatbot is None:
        st.error("Chatbot not initialized. Please select a character and start the chat.")
        return

    # Create a container for the chat content
    chat_container = st.container()

    # Add Restart Conversation button outside the chat container
    if st.button("Restart Conversation"):
        restart_conversation()

    # Use the chat container for all chat content
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Display streamed introduction if it's the first message
        if not st.session_state.messages:
            if st.session_state.chat_mode == "Single Character":
                with st.chat_message("assistant"):
                    intro = st.write_stream(st.session_state.chatbot.start_conversation())
                st.session_state.messages.append({"role": "assistant", "content": intro})
            else:
                st.session_state.chatbot.start_conversation()
    
    # Chat input
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
            # Get character responses
            if st.session_state.chat_mode == "Single Character":
                with st.chat_message("assistant"):
                    response_generator, summarized = st.session_state.chatbot.get_response(prompt)
                    response = st.write_stream(response_generator)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                responses = st.session_state.chatbot.get_responses(prompt)
            
                # Character discussion
                if st.session_state.use_discussion:
                    st.markdown("### Character Discussion")
                    discussion_responses = st.session_state.chatbot.get_discussion_responses(responses)
        
            # Optional: Add moderator summary for multiple characters if enabled
            if st.session_state.chat_mode == "Multiple Characters" and len(st.session_state.selected_characters) > 1 and st.session_state.use_moderator:
                with st.spinner("Moderator is summarizing..."):
                    st.session_state.chatbot.get_moderator_response(discussion_responses if st.session_state.use_discussion else responses)

def home_page():
    st.write("# Welcome to the Expert Chat System! 🧠🤖")
    
    st.markdown("""
    This app allows you to create and chat with AI-powered experts. Whether you're seeking professional advice, 
    exploring different fields of knowledge, or just looking for insightful conversations, this system is for you!

    ## How to Use the App

    1. **Create an Expert**
        - Click on 'Create New Expert' in the sidebar.
        - Enter the expert's field and specialization.
        - Optionally upload an example file for more context.
        - Your new expert will be ready for consultation!

    2. **Choose Your Chat Mode**
        - Select 'Single Expert' to consult with one expert at a time.
        - Choose 'Multiple Experts' to engage in group discussions.

    3. **Start Chatting**
        - In Single Expert mode, select one expert and start the chat.
        - In Multiple Experts mode, select two or more experts for a panel discussion.
        - Type your questions and watch the experts respond!

    4. **Special Features**
        - In Multiple Experts mode, enable 'Experts Discussion' to let experts interact with each other.
        - Use the 'Reset Chat' button to clear the current conversation and return to the main menu.

    ## Tips for a Great Experience
    - Be specific when creating your expert! The more detailed the field and specialization, the more focused the advice.
    - In Multiple Experts mode, try combining experts from different fields for interdisciplinary insights.
    - Don't hesitate to reset the chat and try different combinations of experts.

    ## Privacy Note
    - The developer is not seeing and storing your conversation.
    - Data is not used to train models.
    - Data is used only for abuse monitoring and will be deleted in 30 days, by OpenAI.
    
    Ready to start? Create a new expert or choose a chat mode from the sidebar!
    """)

    st.info("👈 Use the sidebar to navigate through different features of the app!")

def manage_prompts():
  st.subheader("Manage Character Prompts")

  prompts = list_existing_prompts()
  selected_prompt = st.selectbox("Select a prompt to manage:", prompts)

  if selected_prompt:
      prompt_path = f"./few_shot_prompts/{selected_prompt}"
      with open(prompt_path, 'r') as file:
          prompt_content = yaml.safe_load(file)['character_prompt']

      st.text_area("Prompt Content:", value=prompt_content, height=300, key="prompt_editor")

      col1, col2 = st.columns(2)
      with col1:
          if st.button("Save Changes"):
              updated_content = st.session_state.prompt_editor
              with open(prompt_path, 'w') as file:
                  yaml.dump({"character_prompt": updated_content}, file)
              st.success("Prompt updated successfully!")

      with col2:
          if "delete_confirmation" not in st.session_state:
              st.session_state.delete_confirmation = False

          if st.session_state.delete_confirmation:
              if st.button("Confirm Delete"):
                  os.remove(prompt_path)
                  st.success(f"Prompt {selected_prompt} deleted successfully!")
                  st.session_state.delete_confirmation = False
                  time.sleep(2)  # Give user time to see the success message
                  st.rerun()
              if st.button("Cancel"):
                  st.session_state.delete_confirmation = False
                  st.rerun()
          else:
              if st.button("Delete Prompt"):
                  st.session_state.delete_confirmation = True
                  st.rerun()
  else:
      st.info("No prompts available. Create a new character to generate prompts.")   
def main():
    st.title("Expert Chat System 🤖💬")

    initialize_session_state()

    # Sidebar for mode selection and character management
    with st.sidebar:
        st.title("Navigation")
        
        # Home button
        if st.button("🏠 Home"):
            st.session_state.app_mode = "home"
            st.rerun()

        # New option for creating a character
        if st.button("Create New Expert", key="sidebar_create_new_character_button"):
            st.session_state.app_mode = "create_character"
            st.rerun()

        # Create a chat mode selection
        if st.button("Start Chatting", key="start_chat_button"):
            st.session_state.app_mode = "select_characters"
            st.session_state.selected_characters = []
            st.rerun()
            
        # Button for managing prompts
        if st.button("Manage Prompts", key="manage_prompts_button"):
            st.session_state.app_mode = "manage_prompts"
            st.rerun()
            

        if st.session_state.app_mode == "select_characters" or st.session_state.app_mode == "chat":
            new_chat_mode = st.radio(
                "Choose your chat mode:",
                ["Single Expert", "Multiple Experts"],
                key="chat_mode_radio"
            )
            
            # Check if the chat mode has changed
            if st.session_state.previous_chat_mode != new_chat_mode:
                st.session_state.previous_chat_mode = new_chat_mode
                st.session_state.chat_mode = new_chat_mode
                reset_chat()
                st.rerun()

            if st.session_state.chat_mode == "Multiple Experts":
                st.session_state.use_discussion = st.checkbox("Enable Experts Discussion", value=st.session_state.use_discussion)
                st.caption("When enabled, experts will respond to each other's comments, creating a more dynamic conversation.")

        if st.button("Reset Chat", key="reset_chat_button"):
            reset_chat()
            st.rerun()

    # Main content area
    if st.session_state.app_mode == "home":
        home_page()
    elif st.session_state.app_mode == "create_character":
        create_new_character()
    elif st.session_state.app_mode == "select_characters":
        select_characters()
    elif st.session_state.app_mode == "chat":
        chat_interface()
    elif st.session_state.app_mode == "manage_prompts":
        manage_prompts()
    else:
        home_page()  # Default to home page if no mode is set


    # Add instructions for the current mode
    if st.session_state.app_mode == "create_character":
        st.info("""
        Creating a New Character:
        1. Enter the character type (e.g., 'Russian Military Expert', 'Steve Jobs').
        2. Specify the goal for this character (e.g., 'understand Russian perspective', 'get advice').
        3. Optionally upload an example file to provide more context.
        4. Click 'Create Character' to generate the character. Then go to "Start Chatting" to begin a conversation.
        5. Or click 'Create Character and Start Chat' to create and start chatting with your character right away.
        """)
    elif st.session_state.app_mode == "select_characters":
        if st.session_state.chat_mode == "Single Character":
            st.info(f"""- Select a character from the dropdown
                      \n- Or type character's name to search
                      \n- Then click 'Start Chat' to begin your conversation.""")
    elif st.session_state.app_mode == "manage_prompts":
        st.info("""
        Managing Prompts:
        1. Select a prompt from the dropdown to view its content.
        2. Edit the prompt in the text area if needed.
        3. Click 'Save Changes' to update the prompt.
        4. Use 'Delete Prompt' to remove a prompt (this action cannot be undone).
        """)
    else:
        # st.info("Select two or more characters from the multi-select dropdown and click 'Start Chat' to begin your group conversation.")
        pass
if __name__ == "__main__":
    main()