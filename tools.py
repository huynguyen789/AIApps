
import os
from typing import List
# 👋 Make sure you've also installed the OpenAI SDK through: pip install openai
from openai import OpenAI
from toolhouse import Toolhouse

# Let's set our API Keys.
# Please remember to use a safer system to store your API KEYS 
# after finishing the quick start.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
th = Toolhouse(api_key="th-89SKfDUPdKrvV2WXEZ4GotSy69NJu2_JudE2o6F5_pY",
provider="openai")

# Define the OpenAI model we want to use
MODEL = 'gpt-4o'

messages = [{
    "role": "user",
    "content":
        "Generate FizzBuzz code."
        "Execute it to show me the results up to 10, using the available tools."
}]

response = client.chat.completions.create(
  model=MODEL,
  messages=messages,
  # Passes Code Execution as a tool
  tools=th.get_tools(),
  tool_choice="auto"
)

# Runs the Code Execution tool, gets the result, 
# and appends it to the context
messages += th.run_tools(response)

response = client.chat.completions.create(
  model=MODEL,
  messages=messages,
  tools=th.get_tools()
)
# Prints the response with the answer
print(response.choices[0].message.content)





# # ✋ You need to install dependencies with this command before running the code:
# # 👉 pip install toolhouse anthropic
# from toolhouse import Toolhouse
# from anthropic import Anthropic
# from dotenv import load_dotenv

# load_dotenv()
# # 👉 Notice your API KEY is in plain text here, make sure to not commit it
# th = Toolhouse(provider="anthropic", api_key="th-89SKfDUPdKrvV2WXEZ4GotSy69NJu2_JudE2o6F5_pY")
# # 👉 You need to have the anthropic package installed to use this LLM API
# client = Anthropic()

# def llm_call(messages: list[dict]):
#   return client.messages.create(
#     model="claude-3-5-sonnet-latest",
#     system="Respond directly, do not preface or end your responses with anything.",
#     max_tokens=1000,
#     messages=messages,
#     tools=th.get_tools(),
#   )

# messages = [
#   {"role": "user", "content": """Get the contents of https://platform.openai.com/docs/quickstart?language-preference=python and summarize its key value propositions in three bullet points. 
#    And code example of how to call a model in python"""},
# ]

# response = llm_call(messages)
# print(f"response: {response}")
# messages += th.run_tools(response, append=True)
# final_response = llm_call(messages)
# print(final_response.content[0].text)


