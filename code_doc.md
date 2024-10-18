
Default models for
each provider:

* Openai: gpt-4o
* Anthropic:
  claude-3-5-sonnet-20240620
* Google: gemini-1.5-pro

#set up and call
multiple modesl from different provider:

## Overview

This
module provides a flexible AI-powered search assistant that can use multiple
language models (LLMs) to generate responses based on web search results. It
supports streaming responses for improved user experience.

## Key Components

### 1. Model Setup

The `setup_model(model_name: str)` function initializes the chosen AI
model.

Supported
models:

- Gemini:
  "gemini-pro" (gemini-1.5-pro-002) or "gemini-flash"
  (gemini-1.5-flash-002)
- GPT-4:
  "gpt4-o" (gpt-4o) or "gpt4-o-mini" (gpt-4o-mini)
- Claude:
  "claude" (claude-3-5-sonnet-20240620)

### 2. Response Generation

The `generate_response(model, prompt)` function handles streaming
responses for all supported models.

## Usage

1. Select the AI model
   using the dropdown in the Streamlit interface.
2. Enter a search
   query.
3. Choose the search
   type (fast or deep).
4. Click "Search
   and Summarize" to initiate the process.

## Streaming Responses

The application uses streaming to display partial results
as they become available. This is implemented in the `generate_response`
function:

```python



async def generate_response(model, prompt):



    if isinstance(model, genai.GenerativeModel):  # Gemini



        response =
model.generate_content(prompt, stream=True)



        for chunk in response:



            if chunk.text:



                yield chunk.text



    elif isinstance(model, AsyncAnthropic): 
# Claude



        async with
model.messages.stream(...) as stream:



            async for text in stream.text_stream:



                yield text



    elif isinstance(model, tuple) and isinstance(model[0], AsyncOpenAI):  # GPT-4



        stream = await
client.chat.completions.create(..., stream=True)



        async for chunk in stream:



            if chunk.choices[0].delta.content is not None:



                yield chunk.choices[0].delta.content



```

To use
streaming in your Streamlit app:

```python



response_container = st.empty()



full_response = ""



async for content in
generate_response(model, prompt):



    full_response += content



    response_container.markdown(full_response)



```

## Reusability

The `setup_model` and `generate_response` functions can be easily imported and used in other parts
of the application or in different projects.

Example:

```python



from your_module import setup_model,
generate_response



async def custom_function():



    model = setup_model("gemini-pro")



    prompt = "Your custom
prompt"



    async for content in
generate_response(model, prompt):



        # Process streamed content



        print(content)



```

1. Streamlit Integration

Managing Streamlit Session: A
Comprehensive Guide

Streaming Responses

For
long-running tasks, use streaming to display partial results:

```python



async
def stream_response(prompt):



    response = await
openai_client.chat.completions.create(



        model="gpt-4o",



        messages=[{"role":
"user", "content": prompt}],



        stream=True,



    )



  



    full_content = ""



    placeholder = st.empty()



    async for chunk in response:



        if chunk.choices[0].delta.content:



            full_content +=
chunk.choices[0].delta.content



            placeholder.markdown(full_content)



    return full_content



```

GOOGLE
GEMINI:

Using the Gemini API

This
documentation outlines how to set up and use the Gemini API based on the
provided code.

Setup

Import the necessary libraries:

```python



import
google.generativeai as genai



```

Configure the API with your key:

```python



api_key
= "YOUR_API_KEY_HERE"



genai.configure(api_key=api_key)



```

Creating a Model

Define the generation config:

```python



generation_config
= {



    "temperature": 0,



    "max_output_tokens": 8192,



}



```

Set up safety settings:

```python



safety_settings
= [



    {



        "category":
"HARM_CATEGORY_DANGEROUS",



        "threshold":
"BLOCK_NONE",



    },



    {



        "category":
"HARM_CATEGORY_HARASSMENT",



        "threshold":
"BLOCK_NONE",



    },



    {



        "category":
"HARM_CATEGORY_HATE_SPEECH",



        "threshold":
"BLOCK_NONE",



    },



    {



        "category":
"HARM_CATEGORY_SEXUALLY_EXPLICIT",



        "threshold":
"BLOCK_NONE",



    },



    {



        "category":
"HARM_CATEGORY_DANGEROUS_CONTENT",



        "threshold":
"BLOCK_NONE",



    }



]



```

Create the model:

```python



model
= genai.GenerativeModel(



    model_name="gemini-1.5-flash",



    generation_config=generation_config,



    system_instruction=system_instruction,



    safety_settings=safety_settings



)



```

Generating Content

To
generate content using the model:

```python



response
= model.generate_content(prompt, stream=True)



 



for
chunk in response:



    if chunk.candidates:



        candidate = chunk.candidates[0]



        if candidate.content and
candidate.content.parts:



            content =
candidate.content.parts[0].text



            # Process or display the content



```

Handling Safety Filters

The
code includes handling for safety filters:

```python



if
candidate.finish_reason == "SAFETY":



    safety_message = "\n\nNote: The
response was filtered due to safety concerns.\nSafety ratings:\n"



    for rating in candidate.safety_ratings:



        safety_message += f"- Category:
{rating.category}, Probability: {rating.probability}\n"



    # Process or display the safety message



```

Retrieving Usage Metadata

After
processing all chunks, you can retrieve usage metadata:

```python



if
hasattr(response, 'usage_metadata'):



    prompt_tokens =
response.usage_metadata.prompt_token_count



    candidates_tokens =
response.usage_metadata.candidates_token_count



    # Use these values as needed



```

Error Handling

The
code includes basic error handling:

```python



try:



    # API call and processing



except
Exception as e:



    error_message = f"An error occurred:
{e}"



    # Handle or display the error message



```

Remember
to replace `"YOUR_API_KEY_HERE"` with your actual Gemini API key and
adjust the `system_instruction` and other parameters as needed for your
specific use case.

ANTHROPIC:

Anthropic (Claude)

```python



from
anthropic import AsyncAnthropic



 



anthropic_client
= AsyncAnthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])



 



async
def get_anthropic_response(prompt):



    response = await
anthropic_client.messages.create(



     
model="claude-3-sonnet-20240229",  # Use Sonnet as default



        max_tokens=4000,



        temperature=0,



        messages=[



            {



                "role":
"user",



                "content": prompt



            }



        ]



    )



    return response.content[0].text



```

import
anthropic

import
asyncio

async
def stream_anthropic_sonnet():

    api_key =
"your_anthropic_api_key_here"

    async with
anthropic.AsyncClient(api_key=api_key) as aclient:

    async with aclient.messages.stream(

model="claude-3-sonnet-20240229",

    max_tokens=1000,

    temperature=0,

    messages=[

    {

    "role":
"user",

    "content":
"Write a short story about a robot learning to paint."

    }

    ]

    ) as stream:

    async for text in
stream.text_stream:

    print(text, end="",
flush=True)

    print()  # New line after completion

asyncio.run(stream_anthropic_sonnet())

OpenAI

#how to call o1  models:

from openai import OpenAI

import os

from dotenv import load_dotenv

load_dotenv()

def test_o1_model():

    client =
OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:

    response =
client.chat.completions.create(

    model="o1-preview",

    messages=[

    {"role": "user", "content": "Say 'Hello, O1!' if you can hear me."}

    ],

    )

    return
response.choices[0].message.content

    except Exception as e:

    return f"Error
occurred: {str(e)}"

response = test_o1_model()

print("Response from O1 model:")

print(response)

Note: This model cannot use async, system prompt, or max token.
Use the exact format api format like above, can change the function format.

<how to use swarm framework agents>

Swarm (experimental, educational)

An educational framework exploring ergonomic, lightweight
multi-agent orchestration. Managed by OpenAI Solution team.

Warning

Swarm is currently an experimental sample framework intended to
explore ergonomic interfaces for multi-agent systems. It is not intended to be
used in production, and therefore has no official support. (This also means we
will not be reviewing PRs or issues!)

The primary goal of Swarm is to showcase the handoff
& routines patterns explored in the [Orchestrating
Agents: Handoffs &amp; Routines](https://cookbook.openai.com/examples/orchestrating_agents) cookbook.
It is not meant as a standalone library, and is primarily for educational
purposes.

Install

Requires Python 3.10+

pip install git+ssh://git@github.com/openai/swarm.git

or

pip install git+https://github.com/openai/swarm.git

Usage

from swarm import Swarm, Agent

client = Swarm()

def transfer_to_agent_b():

    return agent_b

agent_a = Agent(

    name="Agent A",

    instructions="You are a helpful
agent.",

    functions=[transfer_to_agent_b],

)

agent_b = Agent(

    name="Agent B",

    instructions="Only speak in
Haikus.",

)

response = client.run(

    agent=agent_a,

    messages=[{"role":
"user", "content": "I want to talk to agent
B."}],

)

print(response.messages[-1]["content"])

Hope glimmers brightly,

New paths converge gracefully,

What can I assist?

Table of Contents

* [Overview](https://github.com/openai/swarm#overview)
* [Examples](https://github.com/openai/swarm#examples)
* [Documentation](https://github.com/openai/swarm#documentation)
  * [Running Swarm](https://github.com/openai/swarm#running-swarm)
  * [Agents](https://github.com/openai/swarm#agents)
  * [Functions](https://github.com/openai/swarm#functions)
  * [Streaming](https://github.com/openai/swarm#streaming)
* [Evaluations](https://github.com/openai/swarm#evaluations)
* [Utils](https://github.com/openai/swarm#utils)

Overview

Swarm focuses on making agent coordination and execution lightweight,
highly controllable, and easily testable.

It accomplishes this through two primitive
abstractions: Agents and handoffs.
An Agentencompasses instructions and tools, and can at any
point choose to hand off a conversation to another Agent.

These primitives are powerful enough to express rich dynamics
between tools and networks of agents, allowing you to build scalable,
real-world solutions while avoiding a steep learning curve.

Note

Swarm Agents are not related to Assistants in the Assistants
API. They are named similarly for convenience, but are otherwise completely
unrelated. Swarm is entirely powered by the Chat Completions API and is hence
stateless between calls.

Why Swarm

Swarm explores patterns that are lightweight, scalable, and
highly customizable by design. Approaches similar to Swarm are best suited for
situations dealing with a large number of independent capabilities and
instructions that are difficult to encode into a single prompt.

The Assistants API is a great option for developers looking for
fully-hosted threads and built in memory management and retrieval. However,
Swarm is an educational resource for developers curious to learn about
multi-agent orchestration. Swarm runs (almost) entirely on the client and, much
like the Chat Completions API, does not store state between calls.

Examples

Check out /examples for inspiration! Learn more about
each one in its README.

* [basic](https://github.com/openai/swarm/blob/main/examples/basic): Simple
  examples of fundamentals like setup, function calling, handoffs, and
  context variables
* [triage_agent](https://github.com/openai/swarm/blob/main/examples/triage_agent): Simple
  example of setting up a basic triage step to hand off to the right agent
* [weather_agent](https://github.com/openai/swarm/blob/main/examples/weather_agent): Simple
  example of function calling
* [airline](https://github.com/openai/swarm/blob/main/examples/airline): A multi-agent
  setup for handling different customer service requests in an airline
  context.
* [support_bot](https://github.com/openai/swarm/blob/main/examples/support_bot): A customer
  service bot which includes a user interface agent and a help center agent
  with several tools
* [personal_shopper](https://github.com/openai/swarm/blob/main/examples/personal_shopper): A personal
  shopping agent that can help with making sales and refunding orders

Documentation

Running Swarm

Start by instantiating a Swarm client (which internally just
instantiates an OpenAI client).

from swarm import Swarm

client = Swarm()

client.run()

Swarm's run() function is analogous to
the chat.completions.create() function in the Chat Completions API –
it takes messages and returns messages and saves no state
between calls. Importantly, however, it also handles Agent function execution,
hand-offs, context variable references, and can take multiple turns before
returning to the user.

At its core, Swarm's client.run() implements the
following loop:

1. Get
   a completion from the current Agent
2. Execute tool calls and append
   results
3. Switch Agent if necessary
4. Update context variables, if
   necessary
5. If no new function calls,
   return

Arguments

| Argument                                                                                               | Type         | Description                                  | Default |
| ------------------------------------------------------------------------------------------------------ | ------------ | -------------------------------------------- | ------- |
| agent                                                                                                  | Agent        | The                                          |         |
| (initial) agent to be called.                                                                          | (required)   |                                              |         |
| messages                                                                                               | List         | A list of message objects, identical to[Chat |         |
| Completions messages](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages) | (required)   |                                              |         |
| context_variables                                                                                      | dict         | A                                            |         |
| dictionary of additional context variables, available to functions and Agent                           |              |                                              |         |
| instructions                                                                                           | {}           |                                              |         |
| max_turns                                                                                              | int          | The                                          |         |
| maximum number of conversational turns allowed                                                         | float("inf") |                                              |         |
| model_override                                                                                         | str          | An                                           |         |
| optional string to override the model being used by an Agent                                           | None         |                                              |         |
| execute_tools                                                                                          | bool         | If False,                                    |         |
| interrupt execution and immediately returns tool_callsmessage when an                                  |              |                                              |         |
| Agent tries to call a function                                                                         | True         |                                              |         |
| stream                                                                                                 | bool         | If True,                                     |         |
| enables streaming responses                                                                            | False        |                                              |         |
| debug                                                                                                  | bool         | If True,                                     |         |
| enables debug logging                                                                                  | False        |                                              |         |

Once client.run() is finished (after potentially
multiple calls to agents and tools) it will return
a Response containing all the relevant updated state. Specifically,
the new messages, the last Agent to be called, and the most
up-to-date context_variables. You can pass these values (plus new user
messages) in to your next execution of client.run() to continue the
interaction where it left off – much like chat.completions.create().
(The run_demo_loopfunction implements an example of a full execution loop
in /swarm/repl/repl.py.)

Response Fields

| Field                                                                                                            | Type  | Description                                    |
| ---------------------------------------------------------------------------------------------------------------- | ----- | ---------------------------------------------- |
| messages                                                                                                         | List  | A list of message objects generated during the |
| conversation. Very similar to[Chat                                                                               |       |                                                |
| Completions messages](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages), but with |       |                                                |
| a sender field indicating which Agent the message                                                                |       |                                                |
| originated from.                                                                                                 |       |                                                |
| agent                                                                                                            | Agent | The                                            |
| last agent to handle a message.                                                                                  |       |                                                |
| context_variables                                                                                                | dict  | The                                            |
| same as the input variables, plus any changes.                                                                   |       |                                                |

Agents

An Agent simply encapsulates a set
of instructions with a set of functions (plus some
additional settings below), and has the capability to hand off execution to
another Agent.

While it's tempting to personify an Agent as
"someone who does X", it can also be used to represent a very
specific workflow or step defined by a set
of instructions and functions(e.g. a set of steps, a complex
retrieval, single step of data transformation, etc). This allows Agents to
be composed into a network of "agents", "workflows", and
"tasks", all represented by the same primitive.

Agent Fields

| Field                                                            | Type          | Description | Default |
| ---------------------------------------------------------------- | ------------- | ----------- | ------- |
| name                                                             | str           | The         |         |
| name of the agent.                                               | "Agent"       |             |         |
| model                                                            | str           | The         |         |
| model to be used by the agent.                                   | "gpt-4o"      |             |         |
| instructions                                                     | str or func() |             |         |
| -> str                                                           | Instructions  |             |         |
| for the agent, can be a string or a callable returning a string. | "You          |             |         |
| are a helpful agent."                                            |               |             |         |
| functions                                                        | List          | A           |         |
| list of functions that the agent can call.                       | []            |             |         |
| tool_choice                                                      | str           | The         |         |
| tool choice for the agent, if any.                               | None          |             |         |

Instructions

Agent instructions are directly converted into
the system prompt of a conversation (as the first message). Only
the instructions of the active Agent will be present at any
given time (e.g. if there is an Agent handoff,
the system prompt will change, but the chat history will not.)

agent = Agent(

   instructions="You are a helpful
agent."

)

The instructions can either be a regular str, or
a function that returns a str. The function can optionally receive
a context_variables parameter, which will be populated by
the context_variables passed into client.run().

def instructions(context_variables):

   user_name =
context_variables["user_name"]

   return f"Help the user,
{user_name}, do whatever they want."

agent = Agent(

instructions=instructions

)

response = client.run(

   agent=agent,

   messages=[{"role":"user",
"content": "Hi!"}],

context_variables={"user_name":"John"}

)

print(response.messages[-1]["content"])

Hi John, how can I assist you today?

Functions

* Swarm Agents can call
  python functions directly.
* Function should usually
  return a str (values will be attempted to be cast as
  a str).
* If a function returns
  an Agent, execution will be transfered to that Agent.
* If a function defines
  a context_variables parameter, it will be populated by
  the context_variables passed into client.run().

def greet(context_variables, language):

   user_name =
context_variables["user_name"]

   greeting = "Hola" if
language.lower() == "spanish" else "Hello"

   print(f"{greeting},
{user_name}!")

   return "Done"

agent = Agent(

   functions=[print_hello]

)

client.run(

   agent=agent,

   messages=[{"role":
"user", "content": "Usa greet() por
favor."}],

context_variables={"user_name": "John"}

)

Hola, John!

* If
  an Agent function call has an error (missing function, wrong
  argument, error) an error response will be appended to the chat so
  the Agent can recover gracefully.
* If multiple functions are
  called by the Agent, they will be executed in that order.

Handoffs and Updating Context
Variables

An Agent can hand off to another Agent by
returning it in a function.

sales_agent = Agent(name="Sales Agent")

def transfer_to_sales():

   return sales_agent

agent = Agent(functions=[transfer_to_sales])

response = client.run(agent,
[{"role":"user", "content":"Transfer me to
sales."}])

print(response.agent.name)

Sales Agent

It can also update the context_variables by returning
a more complete Result object. This can also contain
a value and an agent, in case you want a single function to
return a value, update the agent, and update the context variables (or any
subset of the three).

sales_agent = Agent(name="Sales Agent")

def talk_to_sales():

   print("Hello,
World!")

   return Result(

    value="Done",

    agent=sales_agent,

context_variables={"department": "sales"}

   )

agent = Agent(functions=[talk_to_sales])

response = client.run(

   agent=agent,

   messages=[{"role":
"user", "content": "Transfer me to
sales"}],

context_variables={"user_name":
"John"}

)

print(response.agent.name)

print(response.context_variables)

Sales Agent

{'department': 'sales', 'user_name': 'John'}

Note

If an Agent calls multiple functions to hand-off to
an Agent, only the last handoff function will be used.

Function Schemas

Swarm automatically converts functions into a JSON Schema that
is passed into Chat Completions tools.

* Docstrings are turned into
  the function description.
* Parameters without default
  values are set to required.
* Type hints are mapped to the
  parameter's type (and default to string).
* Per-parameter descriptions
  are not explicitly supported, but should work similarly if just added in
  the docstring. (In the future docstring argument parsing may be added.)

def greet(name, age: int, location: str = "New
York"):

   """Greets the user.
Make sure to get their name and age before calling.

   Args:

    name: Name of the user.

    age: Age of the user.

    location: Best place on
earth.

   """

   print(f"Hello {name}, glad you
are {age} in {location}!")

{

   "type":
"function",

   "function": {

    "name":
"greet",

    "description":
"Greets the user. Make sure to get their name and age before
calling.\n\nArgs:\n   name: Name of the
user.\n   age: Age of the user.\n   location: Best place on
earth.",

    "parameters":
{

    "type":
"object",

    "properties":
{

    "name":
{"type": "string"},

    "age":
{"type": "integer"},

    "location":
{"type": "string"}

    },

    "required":
["name", "age"]

    }

   }

}

Streaming

stream = client.run(agent, messages, stream=True)

for chunk in stream:

   print(chunk)

Uses the same events as [Chat
Completions API streaming](https://platform.openai.com/docs/api-reference/streaming).
See process_and_print_streaming_response in /swarm/repl/repl.py as
an example.

Two new event types have been added:

* {"delim":"start"} and {"delim":"start"},
  to signal each time an Agent handles a single message (response
  or function call). This helps identify switches between Agents.
* {"response":
  Response} will return a Response object at the end of a
  stream with the aggregated (complete) response, for convenience.

Evaluations

Evaluations are crucial to any project, and we encourage
developers to bring their own eval suites to test the performance of their
swarms. For reference, we have some examples for how to eval swarm in
the airline, weather_agent and triage_agent quickstart
examples. See the READMEs for more details.

Utils

Use the run_demo_loop to test out your swarm! This
will run a REPL on your command line. Supports streaming.

from swarm.repl import
run_demo_loop

...

run_demo_loop(agent, stream=True)

</how to use swarm framework agents>

#call
models

```python



from
openai import AsyncOpenAI



 



openai_client
= AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])



 



async
def get_openai_response(prompt):



    response = await
openai_client.chat.completions.create(



        model="gpt-4o",  # Use GPT-4o as default



        messages=[



            {"role":
"system", "content": "You are a helpful
assistant."},



            {"role":
"user", "content": prompt}



        ],



        temperature=0



    )



    return response.choices[0].message.content



```

3. Function Calling

Function
calling allows models to generate function arguments that adhere to provided
specifications.

Defining Functions

```python



tools
= [



    {



        "type": "function",



        "function": {



            "name":
"get_current_weather",



            "description": "Get
the current weather",



            "parameters": {



                "type":
"object",



                "properties": {



                    "location": {



                        "type":
"string",



                     
"description": "The city and state, e.g. San Francisco,
CA",



                    },



                    "format": {



                        "type":
"string",



                        "enum":
["celsius", "fahrenheit"],



                     
"description": "The temperature unit to use. Infer this
from the users location.",



                    },



                },



                "required":
["location", "format"],



            },



        }



    },



    # Add more function definitions here



]



```

Using Functions

```python



response
= client.chat.completions.create(



    model="gpt-4o",



    messages=messages,



    tools=tools,



    tool_choice="auto"  # Let the model decide when to use functions



)



 



#
Check if the model wants to call a function



if
response.choices[0].message.tool_calls:



    # Extract function call details and execute
the function



    tool_call =
response.choices[0].message.tool_calls[0]



    function_name = tool_call.function.name



    function_args =
json.loads(tool_call.function.arguments)



  



    # Execute the function and get the result



    function_response =
globals()[function_name](**function_args)



  



    # Add the function response to the
conversation



    messages.append({



        "role": "function",



        "name": function_name,



        "content": function_response



    })



```

4. Prompt Caching

OpenAI
offers discounted prompt caching for prompts exceeding 1024 tokens, resulting
in up to an 80% reduction in latency for longer prompts over 10,000 tokens.

Key Features

Automatically activates for prompts longer than 1024 tokens

Caching is scoped at the organization level

Eligible for zero data retention

Checking Cached Tokens

```python



response
= client.chat.completions.create(



    model="gpt-4o",



    messages=[...],



    # other parameters



)



 



cached_tokens
= response.usage.prompt_tokens_details.cached_tokens



print(f"Number
of cached tokens: {cached_tokens}")



```

Best Practices

Place static or frequently reused content at the beginning of prompts

Maintain consistent usage patterns to prevent cache evictions

Monitor key metrics like cache hit rates, latency, and proportion of cached
tokens

Caching with Tools and Multi-turn Conversations

Ensure tool definitions and their order remain identical for caching

Append new elements to the end of the messages array for multi-turn
conversations

Caching with Images

Images (linked or base64 encoded) qualify for caching

Keep the `detail` parameter consistent for image tokenization

GPT-4o models add extra tokens for image processing costs

5. Assistants API

The
Assistants API is a stateful evolution of the Chat Completions API, simplifying
the creation of assistant-like experiences and enabling access to tools like
Code Interpreter and Retrieval.

Key Components

**Assistants**: Encapsulate a base model, instructions, tools, and context
documents

**Threads**: Represent the state of a conversation

**Runs**: Power the execution of an Assistant on a Thread, including responses
and tool use

Creating an Assistant

```python



assistant
= client.beta.assistants.create(



    name="Math Tutor",



    instructions="You are a personal math
tutor. Answer questions briefly, in a sentence or less.",



    model="gpt-4o",



   max_token=4000,



    tools=[{"type":
"code_interpreter"}]



)



```

Creating a Thread and Run

```python



thread
= client.beta.threads.create()



 



message
= client.beta.threads.messages.create(



    thread_id=thread.id,



    role="user",



    content="I need to solve the equation
`3x + 11 = 14`. Can you help me?"



)



 



run
= client.beta.threads.runs.create(



    thread_id=thread.id,



    assistant_id=assistant.id



)



 



#
Wait for the run to complete



run
= wait_on_run(run, thread)



 



#
Retrieve messages



messages
= client.beta.threads.messages.list(thread_id=thread.id)



```

6. Best Practices

Use session state for maintaining app state in Streamlit

* When coding  a new app in the AI Assistant apps. Only
  focus on tha

Implement streaming for long-running tasks to improve user experience

Use asynchronous programming for efficient API calls

Implement proper error handling for API calls and user inputs

Place static content at the beginning of prompts for better cache efficiency

Monitor caching metrics to optimize performance and cost-efficiency

When using the Assistants API, leverage tools like Code Interpreter and
Retrieval for enhanced capabilities

For function calling, provide clear and specific function descriptions to guide
the model's usage

* Prompt managements: save the
  prompts into ./prompt folder(create if not exits), then load from there.

UNSTRUCTURE
PDFS:

Using Unstructured API for PDF Processing and Caching

Setup

Install required libraries:

```



   pip install unstructured_client
python-dotenv



```

Set up your Unstructured API key in a `.env` file:

```



   UNSTRUCTURED_API_KEY=your_api_key_here



```

Import necessary modules:

```python



   from unstructured_client import
UnstructuredClient



   from unstructured_client.models import
shared



   from dotenv import load_dotenv



   import os



   import glob



   import json



```

Processing PDFs

Initialize the UnstructuredClient:

```python



   load_dotenv()



   unstructured_api_key =
os.getenv("UNSTRUCTURED_API_KEY")



   s =
UnstructuredClient(api_key_auth=unstructured_api_key,
server_url='https://api.unstructured.io')



```

Process PDFs in a folder:

```python



   def process_pdfs(input_folder,
strategy="auto"):



       combined_content = []



       for filename in
glob.glob(os.path.join(input_folder, "*.pdf")):



           with open(filename, "rb")
as file:



               req =
shared.PartitionParameters(



                   files=shared.Files(



                       content=file.read(),



                       file_name=filename,



                   ),



                   strategy=strategy,



               )



               res = s.general.partition(req)



            
combined_content.extend(res.elements)



       return combined_content



```

Caching Results

Implement caching mechanism:

```python



   def process_pdfs_and_cache(input_folder,
output_folder, strategy="auto"):



       os.makedirs(output_folder,
exist_ok=True)



       folder_name =
os.path.basename(os.path.normpath(input_folder))



       cache_file_path =
os.path.join(output_folder, f'{folder_name}_combined_content.json')



 



       if os.path.exists(cache_file_path):



           with open(cache_file_path, 'r',
encoding='utf-8') as f:



               return json.load(f)



       else:



           combined_content =
process_pdfs(input_folder, strategy)



           with open(cache_file_path, 'w',
encoding='utf-8') as f:



               json.dump(combined_content, f)



           return combined_content



```

Use the cached results:

```python



   combined_content =
process_pdfs_and_cache("path/to/pdf/folder", "./cache")



```

This
setup allows you to process PDFs using the Unstructured API and cache the
results for faster subsequent access. The `process_pdfs_and_cache` function
checks for existing cached results before processing, improving efficiency for
repeated use

Default models for
each provider:

* Openai: gpt-4o
* Anthropic:
  claude-3-5-sonnet-20240620
* Google: gemini-1.5-pro

1. Streamlit Integration

Managing Streamlit Session: A
Comprehensive Guide

Streaming Responses

For
long-running tasks, use streaming to display partial results:

```python



async
def stream_response(prompt):



    response = await
openai_client.chat.completions.create(



        model="gpt-4o",



        messages=[{"role":
"user", "content": prompt}],



        stream=True,



    )



  



    full_content = ""



    placeholder = st.empty()



    async for chunk in response:



        if chunk.choices[0].delta.content:



            full_content +=
chunk.choices[0].delta.content



            placeholder.markdown(full_content)



    return full_content



```

GOOGLE
GEMINI:

Using the Gemini API

This
documentation outlines how to set up and use the Gemini API based on the
provided code.

Setup

Import the necessary libraries:

```python



import
google.generativeai as genai



```

Configure the API with your key:

```python



api_key
= "YOUR_API_KEY_HERE"



genai.configure(api_key=api_key)



```

Creating a Model

Define the generation config:

```python



generation_config
= {



    "temperature": 0,



    "max_output_tokens": 8192,



}



```

Set up safety settings:

```python



safety_settings
= [



    {



        "category":
"HARM_CATEGORY_DANGEROUS",



        "threshold":
"BLOCK_NONE",



    },



    {



        "category":
"HARM_CATEGORY_HARASSMENT",



        "threshold":
"BLOCK_NONE",



    },



    {



        "category":
"HARM_CATEGORY_HATE_SPEECH",



        "threshold":
"BLOCK_NONE",



    },



    {



        "category":
"HARM_CATEGORY_SEXUALLY_EXPLICIT",



        "threshold":
"BLOCK_NONE",



    },



    {



        "category":
"HARM_CATEGORY_DANGEROUS_CONTENT",



        "threshold":
"BLOCK_NONE",



    }



]



```

Create the model:

```python



model
= genai.GenerativeModel(



    model_name="gemini-1.5-flash",



    generation_config=generation_config,



    system_instruction=system_instruction,



    safety_settings=safety_settings



)



```

Generating Content

To
generate content using the model:

```python



response
= model.generate_content(prompt, stream=True)



 



for
chunk in response:



    if chunk.candidates:



        candidate = chunk.candidates[0]



        if candidate.content and
candidate.content.parts:



            content =
candidate.content.parts[0].text



            # Process or display the content



```

Handling Safety Filters

The
code includes handling for safety filters:

```python



if
candidate.finish_reason == "SAFETY":



    safety_message = "\n\nNote: The
response was filtered due to safety concerns.\nSafety ratings:\n"



    for rating in candidate.safety_ratings:



        safety_message += f"- Category:
{rating.category}, Probability: {rating.probability}\n"



    # Process or display the safety message



```

Retrieving Usage Metadata

After
processing all chunks, you can retrieve usage metadata:

```python



if
hasattr(response, 'usage_metadata'):



    prompt_tokens =
response.usage_metadata.prompt_token_count



    candidates_tokens =
response.usage_metadata.candidates_token_count



    # Use these values as needed



```

Error Handling

The
code includes basic error handling:

```python



try:



    # API call and processing



except
Exception as e:



    error_message = f"An error occurred:
{e}"



    # Handle or display the error message



```

Remember
to replace `"YOUR_API_KEY_HERE"` with your actual Gemini API key and
adjust the `system_instruction` and other parameters as needed for your
specific use case.

ANTHROPIC:

Anthropic (Claude)

```python



from
anthropic import AsyncAnthropic



 



anthropic_client
= AsyncAnthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])



 



async
def get_anthropic_response(prompt):



    response = await
anthropic_client.messages.create(



     
model="claude-3-sonnet-20240229",  # Use Sonnet as default



        max_tokens=4000,



        temperature=0,



        messages=[



            {



                "role":
"user",



                "content": prompt



            }



        ]



    )



    return response.content[0].text



```

import
anthropic

import
asyncio

async
def stream_anthropic_sonnet():

    api_key =
"your_anthropic_api_key_here"

    async with
anthropic.AsyncClient(api_key=api_key) as aclient:

    async with aclient.messages.stream(

model="claude-3-sonnet-20240229",

    max_tokens=1000,

    temperature=0,

    messages=[

    {

    "role":
"user",

    "content":
"Write a short story about a robot learning to paint."

    }

    ]

    ) as stream:

    async for text in
stream.text_stream:

    print(text, end="",
flush=True)

    print()  # New line after completion

asyncio.run(stream_anthropic_sonnet())

OpenAI

#how to call o1  models:

from openai import OpenAI

import os

from dotenv import load_dotenv

load_dotenv()

def test_o1_model():

    client =
OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:

    response =
client.chat.completions.create(

    model="o1-preview",

    messages=[

    {"role": "user", "content": "Say 'Hello, O1!' if you can hear me."}

    ],

    )

    return
response.choices[0].message.content

    except Exception as e:

    return f"Error
occurred: {str(e)}"

response = test_o1_model()

print("Response from O1 model:")

print(response)

Note: This model cannot use async, system prompt, or max token.
Use the exact format api format like above, can change the function format.

<how to use swarm framework agents>

Swarm (experimental, educational)

An educational framework exploring ergonomic, lightweight
multi-agent orchestration. Managed by OpenAI Solution team.

Warning

Swarm is currently an experimental sample framework intended to
explore ergonomic interfaces for multi-agent systems. It is not intended to be
used in production, and therefore has no official support. (This also means we
will not be reviewing PRs or issues!)

The primary goal of Swarm is to showcase the handoff
& routines patterns explored in the [Orchestrating
Agents: Handoffs &amp; Routines](https://cookbook.openai.com/examples/orchestrating_agents) cookbook.
It is not meant as a standalone library, and is primarily for educational
purposes.

Install

Requires Python 3.10+

pip install git+ssh://git@github.com/openai/swarm.git

or

pip install git+https://github.com/openai/swarm.git

Usage

from swarm import Swarm, Agent

client = Swarm()

def transfer_to_agent_b():

    return agent_b

agent_a = Agent(

    name="Agent A",

    instructions="You are a helpful
agent.",

    functions=[transfer_to_agent_b],

)

agent_b = Agent(

    name="Agent B",

    instructions="Only speak in
Haikus.",

)

response = client.run(

    agent=agent_a,

    messages=[{"role":
"user", "content": "I want to talk to agent
B."}],

)

print(response.messages[-1]["content"])

Hope glimmers brightly,

New paths converge gracefully,

What can I assist?

Table of Contents

* [Overview](https://github.com/openai/swarm#overview)
* [Examples](https://github.com/openai/swarm#examples)
* [Documentation](https://github.com/openai/swarm#documentation)
  * [Running Swarm](https://github.com/openai/swarm#running-swarm)
  * [Agents](https://github.com/openai/swarm#agents)
  * [Functions](https://github.com/openai/swarm#functions)
  * [Streaming](https://github.com/openai/swarm#streaming)
* [Evaluations](https://github.com/openai/swarm#evaluations)
* [Utils](https://github.com/openai/swarm#utils)

Overview

Swarm focuses on making agent coordination and execution lightweight,
highly controllable, and easily testable.

It accomplishes this through two primitive
abstractions: Agents and handoffs.
An Agentencompasses instructions and tools, and can at any
point choose to hand off a conversation to another Agent.

These primitives are powerful enough to express rich dynamics
between tools and networks of agents, allowing you to build scalable,
real-world solutions while avoiding a steep learning curve.

Note

Swarm Agents are not related to Assistants in the Assistants
API. They are named similarly for convenience, but are otherwise completely
unrelated. Swarm is entirely powered by the Chat Completions API and is hence
stateless between calls.

Why Swarm

Swarm explores patterns that are lightweight, scalable, and
highly customizable by design. Approaches similar to Swarm are best suited for
situations dealing with a large number of independent capabilities and
instructions that are difficult to encode into a single prompt.

The Assistants API is a great option for developers looking for
fully-hosted threads and built in memory management and retrieval. However,
Swarm is an educational resource for developers curious to learn about
multi-agent orchestration. Swarm runs (almost) entirely on the client and, much
like the Chat Completions API, does not store state between calls.

Examples

Check out /examples for inspiration! Learn more about
each one in its README.

* [basic](https://github.com/openai/swarm/blob/main/examples/basic): Simple
  examples of fundamentals like setup, function calling, handoffs, and
  context variables
* [triage_agent](https://github.com/openai/swarm/blob/main/examples/triage_agent): Simple
  example of setting up a basic triage step to hand off to the right agent
* [weather_agent](https://github.com/openai/swarm/blob/main/examples/weather_agent): Simple
  example of function calling
* [airline](https://github.com/openai/swarm/blob/main/examples/airline): A multi-agent
  setup for handling different customer service requests in an airline
  context.
* [support_bot](https://github.com/openai/swarm/blob/main/examples/support_bot): A customer
  service bot which includes a user interface agent and a help center agent
  with several tools
* [personal_shopper](https://github.com/openai/swarm/blob/main/examples/personal_shopper): A personal
  shopping agent that can help with making sales and refunding orders

Documentation

Running Swarm

Start by instantiating a Swarm client (which internally just
instantiates an OpenAI client).

from swarm import Swarm

client = Swarm()

client.run()

Swarm's run() function is analogous to
the chat.completions.create() function in the Chat Completions API –
it takes messages and returns messages and saves no state
between calls. Importantly, however, it also handles Agent function execution,
hand-offs, context variable references, and can take multiple turns before
returning to the user.

At its core, Swarm's client.run() implements the
following loop:

1. Get
   a completion from the current Agent
2. Execute tool calls and append
   results
3. Switch Agent if necessary
4. Update context variables, if
   necessary
5. If no new function calls,
   return

Arguments

| Argument                                                                                               | Type         | Description                                  | Default |
| ------------------------------------------------------------------------------------------------------ | ------------ | -------------------------------------------- | ------- |
| agent                                                                                                  | Agent        | The                                          |         |
| (initial) agent to be called.                                                                          | (required)   |                                              |         |
| messages                                                                                               | List         | A list of message objects, identical to[Chat |         |
| Completions messages](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages) | (required)   |                                              |         |
| context_variables                                                                                      | dict         | A                                            |         |
| dictionary of additional context variables, available to functions and Agent                           |              |                                              |         |
| instructions                                                                                           | {}           |                                              |         |
| max_turns                                                                                              | int          | The                                          |         |
| maximum number of conversational turns allowed                                                         | float("inf") |                                              |         |
| model_override                                                                                         | str          | An                                           |         |
| optional string to override the model being used by an Agent                                           | None         |                                              |         |
| execute_tools                                                                                          | bool         | If False,                                    |         |
| interrupt execution and immediately returns tool_callsmessage when an                                  |              |                                              |         |
| Agent tries to call a function                                                                         | True         |                                              |         |
| stream                                                                                                 | bool         | If True,                                     |         |
| enables streaming responses                                                                            | False        |                                              |         |
| debug                                                                                                  | bool         | If True,                                     |         |
| enables debug logging                                                                                  | False        |                                              |         |

Once client.run() is finished (after potentially
multiple calls to agents and tools) it will return
a Response containing all the relevant updated state. Specifically,
the new messages, the last Agent to be called, and the most
up-to-date context_variables. You can pass these values (plus new user
messages) in to your next execution of client.run() to continue the
interaction where it left off – much like chat.completions.create().
(The run_demo_loopfunction implements an example of a full execution loop
in /swarm/repl/repl.py.)

Response Fields

| Field                                                                                                            | Type  | Description                                    |
| ---------------------------------------------------------------------------------------------------------------- | ----- | ---------------------------------------------- |
| messages                                                                                                         | List  | A list of message objects generated during the |
| conversation. Very similar to[Chat                                                                               |       |                                                |
| Completions messages](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages), but with |       |                                                |
| a sender field indicating which Agent the message                                                                |       |                                                |
| originated from.                                                                                                 |       |                                                |
| agent                                                                                                            | Agent | The                                            |
| last agent to handle a message.                                                                                  |       |                                                |
| context_variables                                                                                                | dict  | The                                            |
| same as the input variables, plus any changes.                                                                   |       |                                                |

Agents

An Agent simply encapsulates a set
of instructions with a set of functions (plus some
additional settings below), and has the capability to hand off execution to
another Agent.

While it's tempting to personify an Agent as
"someone who does X", it can also be used to represent a very
specific workflow or step defined by a set
of instructions and functions(e.g. a set of steps, a complex
retrieval, single step of data transformation, etc). This allows Agents to
be composed into a network of "agents", "workflows", and
"tasks", all represented by the same primitive.

Agent Fields

| Field                                                            | Type          | Description | Default |
| ---------------------------------------------------------------- | ------------- | ----------- | ------- |
| name                                                             | str           | The         |         |
| name of the agent.                                               | "Agent"       |             |         |
| model                                                            | str           | The         |         |
| model to be used by the agent.                                   | "gpt-4o"      |             |         |
| instructions                                                     | str or func() |             |         |
| -> str                                                           | Instructions  |             |         |
| for the agent, can be a string or a callable returning a string. | "You          |             |         |
| are a helpful agent."                                            |               |             |         |
| functions                                                        | List          | A           |         |
| list of functions that the agent can call.                       | []            |             |         |
| tool_choice                                                      | str           | The         |         |
| tool choice for the agent, if any.                               | None          |             |         |

Instructions

Agent instructions are directly converted into
the system prompt of a conversation (as the first message). Only
the instructions of the active Agent will be present at any
given time (e.g. if there is an Agent handoff,
the system prompt will change, but the chat history will not.)

agent = Agent(

   instructions="You are a helpful
agent."

)

The instructions can either be a regular str, or
a function that returns a str. The function can optionally receive
a context_variables parameter, which will be populated by
the context_variables passed into client.run().

def instructions(context_variables):

   user_name =
context_variables["user_name"]

   return f"Help the user,
{user_name}, do whatever they want."

agent = Agent(

instructions=instructions

)

response = client.run(

   agent=agent,

   messages=[{"role":"user",
"content": "Hi!"}],

context_variables={"user_name":"John"}

)

print(response.messages[-1]["content"])

Hi John, how can I assist you today?

Functions

* Swarm Agents can call
  python functions directly.
* Function should usually
  return a str (values will be attempted to be cast as
  a str).
* If a function returns
  an Agent, execution will be transfered to that Agent.
* If a function defines
  a context_variables parameter, it will be populated by
  the context_variables passed into client.run().

def greet(context_variables, language):

   user_name =
context_variables["user_name"]

   greeting = "Hola" if
language.lower() == "spanish" else "Hello"

   print(f"{greeting},
{user_name}!")

   return "Done"

agent = Agent(

   functions=[print_hello]

)

client.run(

   agent=agent,

   messages=[{"role":
"user", "content": "Usa greet() por
favor."}],

context_variables={"user_name": "John"}

)

Hola, John!

* If
  an Agent function call has an error (missing function, wrong
  argument, error) an error response will be appended to the chat so
  the Agent can recover gracefully.
* If multiple functions are
  called by the Agent, they will be executed in that order.

Handoffs and Updating Context
Variables

An Agent can hand off to another Agent by
returning it in a function.

sales_agent = Agent(name="Sales Agent")

def transfer_to_sales():

   return sales_agent

agent = Agent(functions=[transfer_to_sales])

response = client.run(agent,
[{"role":"user", "content":"Transfer me to
sales."}])

print(response.agent.name)

Sales Agent

It can also update the context_variables by returning
a more complete Result object. This can also contain
a value and an agent, in case you want a single function to
return a value, update the agent, and update the context variables (or any
subset of the three).

sales_agent = Agent(name="Sales Agent")

def talk_to_sales():

   print("Hello,
World!")

   return Result(

    value="Done",

    agent=sales_agent,

context_variables={"department": "sales"}

   )

agent = Agent(functions=[talk_to_sales])

response = client.run(

   agent=agent,

   messages=[{"role":
"user", "content": "Transfer me to
sales"}],

context_variables={"user_name":
"John"}

)

print(response.agent.name)

print(response.context_variables)

Sales Agent

{'department': 'sales', 'user_name': 'John'}

Note

If an Agent calls multiple functions to hand-off to
an Agent, only the last handoff function will be used.

Function Schemas

Swarm automatically converts functions into a JSON Schema that
is passed into Chat Completions tools.

* Docstrings are turned into
  the function description.
* Parameters without default
  values are set to required.
* Type hints are mapped to the
  parameter's type (and default to string).
* Per-parameter descriptions
  are not explicitly supported, but should work similarly if just added in
  the docstring. (In the future docstring argument parsing may be added.)

def greet(name, age: int, location: str = "New
York"):

   """Greets the user.
Make sure to get their name and age before calling.

   Args:

    name: Name of the user.

    age: Age of the user.

    location: Best place on
earth.

   """

   print(f"Hello {name}, glad you
are {age} in {location}!")

{

   "type":
"function",

   "function": {

    "name":
"greet",

    "description":
"Greets the user. Make sure to get their name and age before
calling.\n\nArgs:\n   name: Name of the
user.\n   age: Age of the user.\n   location: Best place on
earth.",

    "parameters":
{

    "type":
"object",

    "properties":
{

    "name":
{"type": "string"},

    "age":
{"type": "integer"},

    "location":
{"type": "string"}

    },

    "required":
["name", "age"]

    }

   }

}

Streaming

stream = client.run(agent, messages, stream=True)

for chunk in stream:

   print(chunk)

Uses the same events as [Chat
Completions API streaming](https://platform.openai.com/docs/api-reference/streaming).
See process_and_print_streaming_response in /swarm/repl/repl.py as
an example.

Two new event types have been added:

* {"delim":"start"} and {"delim":"start"},
  to signal each time an Agent handles a single message (response
  or function call). This helps identify switches between Agents.
* {"response":
  Response} will return a Response object at the end of a
  stream with the aggregated (complete) response, for convenience.

Evaluations

Evaluations are crucial to any project, and we encourage
developers to bring their own eval suites to test the performance of their
swarms. For reference, we have some examples for how to eval swarm in
the airline, weather_agent and triage_agent quickstart
examples. See the READMEs for more details.

Utils

Use the run_demo_loop to test out your swarm! This
will run a REPL on your command line. Supports streaming.

from swarm.repl import
run_demo_loop

...

run_demo_loop(agent, stream=True)

</how to use swarm framework agents>

#call
models

```python



from
openai import AsyncOpenAI



 



openai_client
= AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])



 



async
def get_openai_response(prompt):



    response = await
openai_client.chat.completions.create(



        model="gpt-4o",  # Use GPT-4o as default



        messages=[



            {"role":
"system", "content": "You are a helpful
assistant."},



            {"role":
"user", "content": prompt}



        ],



        temperature=0



    )



    return response.choices[0].message.content



```

3. Function Calling

Function
calling allows models to generate function arguments that adhere to provided
specifications.

Defining Functions

```python



tools
= [



    {



        "type": "function",



        "function": {



            "name":
"get_current_weather",



            "description": "Get
the current weather",



            "parameters": {



                "type":
"object",



                "properties": {



                    "location": {



                        "type":
"string",



                     
"description": "The city and state, e.g. San Francisco,
CA",



                    },



                    "format": {



                        "type":
"string",



                        "enum":
["celsius", "fahrenheit"],



                     
"description": "The temperature unit to use. Infer this
from the users location.",



                    },



                },



                "required":
["location", "format"],



            },



        }



    },



    # Add more function definitions here



]



```

Using Functions

```python



response
= client.chat.completions.create(



    model="gpt-4o",



    messages=messages,



    tools=tools,



    tool_choice="auto"  # Let the model decide when to use functions



)



 



#
Check if the model wants to call a function



if
response.choices[0].message.tool_calls:



    # Extract function call details and execute
the function



    tool_call =
response.choices[0].message.tool_calls[0]



    function_name = tool_call.function.name



    function_args =
json.loads(tool_call.function.arguments)



  



    # Execute the function and get the result



    function_response =
globals()[function_name](**function_args)



  



    # Add the function response to the
conversation



    messages.append({



        "role": "function",



        "name": function_name,



        "content": function_response



    })



```

4. Prompt Caching

OpenAI
offers discounted prompt caching for prompts exceeding 1024 tokens, resulting
in up to an 80% reduction in latency for longer prompts over 10,000 tokens.

Key Features

Automatically activates for prompts longer than 1024 tokens

Caching is scoped at the organization level

Eligible for zero data retention

Checking Cached Tokens

```python



response
= client.chat.completions.create(



    model="gpt-4o",



    messages=[...],



    # other parameters



)



 



cached_tokens
= response.usage.prompt_tokens_details.cached_tokens



print(f"Number
of cached tokens: {cached_tokens}")



```

Best Practices

Place static or frequently reused content at the beginning of prompts

Maintain consistent usage patterns to prevent cache evictions

Monitor key metrics like cache hit rates, latency, and proportion of cached
tokens

Caching with Tools and Multi-turn Conversations

Ensure tool definitions and their order remain identical for caching

Append new elements to the end of the messages array for multi-turn
conversations

Caching with Images

Images (linked or base64 encoded) qualify for caching

Keep the `detail` parameter consistent for image tokenization

GPT-4o models add extra tokens for image processing costs

5. Assistants API

The
Assistants API is a stateful evolution of the Chat Completions API, simplifying
the creation of assistant-like experiences and enabling access to tools like
Code Interpreter and Retrieval.

Key Components

**Assistants**: Encapsulate a base model, instructions, tools, and context
documents

**Threads**: Represent the state of a conversation

**Runs**: Power the execution of an Assistant on a Thread, including responses
and tool use

Creating an Assistant

```python



assistant
= client.beta.assistants.create(



    name="Math Tutor",



    instructions="You are a personal math
tutor. Answer questions briefly, in a sentence or less.",



    model="gpt-4o",



   max_token=4000,



    tools=[{"type":
"code_interpreter"}]



)



```

Creating a Thread and Run

```python



thread
= client.beta.threads.create()



 



message
= client.beta.threads.messages.create(



    thread_id=thread.id,



    role="user",



    content="I need to solve the equation
`3x + 11 = 14`. Can you help me?"



)



 



run
= client.beta.threads.runs.create(



    thread_id=thread.id,



    assistant_id=assistant.id



)



 



#
Wait for the run to complete



run
= wait_on_run(run, thread)



 



#
Retrieve messages



messages
= client.beta.threads.messages.list(thread_id=thread.id)



```

6. Best Practices

Use session state for maintaining app state in Streamlit

* When coding  a new app in the AI Assistant apps. Only
  focus on tha

Implement streaming for long-running tasks to improve user experience

Use asynchronous programming for efficient API calls

Implement proper error handling for API calls and user inputs

Place static content at the beginning of prompts for better cache efficiency

Monitor caching metrics to optimize performance and cost-efficiency

When using the Assistants API, leverage tools like Code Interpreter and
Retrieval for enhanced capabilities

For function calling, provide clear and specific function descriptions to guide
the model's usage

* Prompt managements: save the
  prompts into ./prompt folder(create if not exits), then load from there.

UNSTRUCTURE
PDFS:

Using Unstructured API for PDF Processing and Caching

Setup

Install required libraries:

```



   pip install unstructured_client
python-dotenv



```

Set up your Unstructured API key in a `.env` file:

```



   UNSTRUCTURED_API_KEY=your_api_key_here



```

Import necessary modules:

```python



   from unstructured_client import
UnstructuredClient



   from unstructured_client.models import
shared



   from dotenv import load_dotenv



   import os



   import glob



   import json



```

Processing PDFs

Initialize the UnstructuredClient:

```python



   load_dotenv()



   unstructured_api_key =
os.getenv("UNSTRUCTURED_API_KEY")



   s =
UnstructuredClient(api_key_auth=unstructured_api_key,
server_url='https://api.unstructured.io')



```

Process PDFs in a folder:

```python



   def process_pdfs(input_folder,
strategy="auto"):



       combined_content = []



       for filename in
glob.glob(os.path.join(input_folder, "*.pdf")):



           with open(filename, "rb")
as file:



               req =
shared.PartitionParameters(



                   files=shared.Files(



                       content=file.read(),



                       file_name=filename,



                   ),



                   strategy=strategy,



               )



               res = s.general.partition(req)



            
combined_content.extend(res.elements)



       return combined_content



```

Caching Results

Implement caching mechanism:

```python



   def process_pdfs_and_cache(input_folder,
output_folder, strategy="auto"):



       os.makedirs(output_folder,
exist_ok=True)



       folder_name =
os.path.basename(os.path.normpath(input_folder))



       cache_file_path =
os.path.join(output_folder, f'{folder_name}_combined_content.json')



 



       if os.path.exists(cache_file_path):



           with open(cache_file_path, 'r',
encoding='utf-8') as f:



               return json.load(f)



       else:



           combined_content =
process_pdfs(input_folder, strategy)



           with open(cache_file_path, 'w',
encoding='utf-8') as f:



               json.dump(combined_content, f)



           return combined_content



```

Use the cached results:

```python



   combined_content =
process_pdfs_and_cache("path/to/pdf/folder", "./cache")



```

This
setup allows you to process PDFs using the Unstructured API and cache the
results for faster subsequent access. The `process_pdfs_and_cache` function
checks for existing cached results before processing, improving efficiency for
repeated uses.
