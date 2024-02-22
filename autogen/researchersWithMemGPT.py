#file adapted from autogen repo
#we'll be checking how to attach a GraphDB to MemGPT agents https://memgpt.readme.io/docs/data_sources
import autogen
from openai import OpenAI
import json
import os
import dotenv

from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config

#long context  handling: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_capability_long_context_handling.ipynb
#autogen autobuild agents multi agents: https://github.com/microsoft/autogen/blob/main/notebook/autobuild_agent_library.ipynb
# pre frontal cortex archiotechture: https://arxiv.org/pdf/2310.00194.pdf


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

config_list_gpt4 = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    file_location=".",
    filter_dict={
        "model": ["gpt-4", "gpt-4-1106-preview"],
    },
)

##Construct AGENTS for Autogen

gpt4_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list_gpt4,
    "timeout": 120,
}
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)
engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt4_config,
    system_message="""Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
)
scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=gpt4_config,
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.""",
)
planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
    llm_config=gpt4_config,
)
executor = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "paper",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)
critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
    llm_config=gpt4_config,
)

## Settings for MemGPT AutoGen agent


# create a config for the MemGPT AutoGen agent
config_list_memgpt = [
    {
        "model": "gpt-4",
        "context_window": 8192,
        "preset": "memgpt_chat",  # note: you can change the preset here
        # OpenAI specific
        "model_endpoint_type": "openai",
        "openai_key": config_value,
    },
]
llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}

# there are some additional options to do with how you want the interface to look (more info below)
interface_kwargs = {
    "debug": False,
    "show_inner_thoughts": True,
    "show_function_outputs": False,
}

# then pass the config to the constructor
memgpt_autogen_agent = create_memgpt_autogen_agent_from_config(
    "MemGPT_agent",
    llm_config=llm_config_memgpt,
    system_message=f"Your desired MemGPT persona",
    interface_kwargs=interface_kwargs,
    default_auto_reply="...",
    skip_verify=False,  # NOTE: you should set this to True if you expect your MemGPT AutoGen agent to call a function other than send_message on the first turn
    auto_save=False,  # NOTE: set this to True if you want the MemGPT AutoGen agent to save its internal state after each reply - you can also save manually with .save()
)


## Construct GroupChatManager
groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, scientist, planner, executor, critic], messages=[], max_round=50
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

user_proxy.initiate_chat(
    manager,
    message="""
find papers on LLM applications from arxiv in the last week, create a markdown table of different domains.
""",
)

#if a groupchat needs to be reseted, you can use:
# for agent in groupchat.agents:
    #agent.reset()