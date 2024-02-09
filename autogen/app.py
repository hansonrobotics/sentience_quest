import autogen
from openai import OpenAI
import json
import os
import dotenv

from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.teachability import Teachability

# Load the environment variables from the .env file
dotenv.load_dotenv()

# Access the value of the CONFIG variable
config_value = os.getenv("OAI_OPENAI_KEY_VAR")

# Lista de configuración proporcionada
config_list_ = [
    {
        'model': 'gpt-4',
        'api_key':config_value,
    },
    {
        'model': 'gpt-4-1106-preview',
        'api_key': config_value,
    },
]

# Convertir la lista de configuración a una cadena de texto JSON
json_string= json.dumps(config_list_, indent=4)

os.environ["OAI_CONFIG_LIST"] = json_string

config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    file_location=".",
    filter_dict={
        "model": ["gpt-4", "gpt-4-1106-preview"],
    },
)

llm_config=  {
    "timeout":600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0
}

assistant = autogen.AssistantAgent(
    name = "assistant",
    llm_config = llm_config,
    system_message="Helpful assistant"
)

user_proxy = autogen.UserProxyAgent(
    name = "user_proxy",
    human_input_mode="NEVER", #ALWAYS, TERMINATE (once the task is finished it will ask for feedback), NEVER 
    max_consecutive_auto_reply=10, #time of interaction between agents (assisntant and user proxy)
    is_termination_msg= lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir":"_output",
        "use_docker": "python:3"
        },
    llm_config=llm_config,
    

    system_message="""Reply TERMINATE if the task has been solved at full satisfaction. 
    Otherwise, reply CONTINUE, or the reason why the task is not solved yet"""
)

task = """
Search for the API of some weather service, and write a Python script that uses the API to get the current weather for a given city. And bring me back the weather of the city of Buenos Aires.
"""

user_proxy.initiate_chat(
    assistant, 
    message = task
)