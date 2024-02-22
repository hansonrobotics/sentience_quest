#file adapted from: https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/examples/agent_groupchat.py

from openai import OpenAI
import json
import os
import dotenv

import autogen
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config, load_autogen_memgpt_agent
from memgpt.constants import LLM_MAX_TOKENS, DEFAULT_PRESET


LLM_BACKEND = "openai"
# LLM_BACKEND = "azure"
# LLM_BACKEND = "local"

if LLM_BACKEND == "openai":
    # For demo purposes let's use gpt-4
    model = "gpt-4"

    # Load the environment variables from the .env file
    dotenv.load_dotenv()

    # Access the value of the CONFIG variable
    openai_api_key = os.getenv("OAI_OPENAI_KEY_VAR")
    #print(openai_api_key)

    assert openai_api_key, "You must set OAI_OPENAI_KEY_VAR or set LLM_BACKEND to 'local' to run this example"

    # This config is for AutoGen agents that are not powered by MemGPT
    config_list = [
        {
            "model": model,
            "api_key": openai_api_key,
        }
    ]

    # This config is for AutoGen agents that powered by MemGPT
    config_list_memgpt = [
        {
            "model": model,
            "context_window": LLM_MAX_TOKENS[model],
            "preset": DEFAULT_PRESET,
            "model_wrapper": None,
            # OpenAI specific
            "model_endpoint_type": "openai",
            "model_endpoint": "https://api.openai.com/v1",
            "openai_key": openai_api_key,
        },
    ]

elif LLM_BACKEND == "azure":
    # Make sure that you have access to this deployment/model on your Azure account!
    # If you don't have access to the model, the code will fail
    model = "gpt-4"

    azure_openai_api_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_version = os.getenv("AZURE_OPENAI_VERSION")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    assert (
        azure_openai_api_key is not None and azure_openai_version is not None and azure_openai_endpoint is not None
    ), "Set all the required OpenAI Azure variables (see: https://memgpt.readme.io/docs/endpoints#azure-openai)"

    # This config is for AutoGen agents that are not powered by MemGPT
    config_list = [
        {
            "model": model,
            "api_type": "azure",
            "api_key": azure_openai_api_key,
            "api_version": azure_openai_version,
            # note: on versions of pyautogen < 0.2.0, use "api_base"
            # "api_base": azure_openai_endpoint,
            "base_url": azure_openai_endpoint,
        }
    ]

    # This config is for AutoGen agents that powered by MemGPT
    config_list_memgpt = [
        {
            "model": model,
            "context_window": LLM_MAX_TOKENS[model],
            "preset": DEFAULT_PRESET,
            "model_wrapper": None,
            # Azure specific
            "model_endpoint_type": "azure",
            "azure_key": azure_openai_api_key,
            "azure_endpoint": azure_openai_endpoint,
            "azure_version": azure_openai_version,
        },
    ]

elif LLM_BACKEND == "local":
    # Example using LM Studio on a local machine
    # You will have to change the parameters based on your setup

    # Non-MemGPT agents will still use local LLMs, but they will use the ChatCompletions endpoint
    config_list = [
        {
            "model": "NULL",  # not needed
            # note: on versions of pyautogen < 0.2.0 use "api_base", and also uncomment "api_type"
            # "api_base": "http://localhost:1234/v1",
            # "api_type": "open_ai",
            "base_url": "http://localhost:1234/v1",  # ex. "http://127.0.0.1:5001/v1" if you are using webui, "http://localhost:1234/v1/" if you are using LM Studio
            "api_key": "NULL",  #  not needed
        },
    ]

    # MemGPT-powered agents will also use local LLMs, but they need additional setup (also they use the Completions endpoint)
    config_list_memgpt = [
        {
            "preset": DEFAULT_PRESET,
            "model": None,  # only required for Ollama, see: https://memgpt.readme.io/docs/ollama
            "context_window": 8192,  # the context window of your model (for Mistral 7B-based models, it's likely 8192)
            "model_wrapper": "chatml",  # chatml is the default wrapper
            "model_endpoint_type": "lmstudio",  # can use webui, ollama, llamacpp, etc.
            "model_endpoint": "http://localhost:1234",  # the IP address of your LLM backend
        },
    ]

else:
    raise ValueError(LLM_BACKEND)

# If USE_MEMGPT is False, then this example will be the same as the official AutoGen repo
# (https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb)
# If USE_MEMGPT is True, then we swap out the "coder" agent with a MemGPT agent
USE_MEMGPT = True

# Set to True if you want to print MemGPT's inner workings.
DEBUG = False

interface_kwargs = {
    "debug": DEBUG,
    "show_inner_thoughts": True,
    "show_function_outputs": DEBUG,
}

llm_config = {"config_list": config_list, "seed": 42}
llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}

# The user agent
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat", "use_docker": "python:3"},
    human_input_mode="TERMINATE",  # needed?
    default_auto_reply="...",  # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
)

# The agent playing the role of the product manager (PM)
pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config=llm_config,
    default_auto_reply="...",  # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
)

if not USE_MEMGPT:
    # In the AutoGen example, we create an AssistantAgent to play the role of the coder
    coder = autogen.AssistantAgent(
        name="Coder",
        llm_config=llm_config,
    )

else:
    # In our example, we swap this AutoGen agent with a MemGPT agent
    # This MemGPT agent will have all the benefits of MemGPT, ie persistent memory, etc.
    coder = create_memgpt_autogen_agent_from_config(
        "MemGPT_coder_v4",
        llm_config=llm_config_memgpt,
        system_message=f"I am a 10x engineer, trained in Python. I was the first engineer at Uber "
        f"(which I make sure to tell everyone I work with).\n"
        f"You are participating in a group chat with a user ({user_proxy.name}) "
        f"and a product manager ({pm.name}).",
        interface_kwargs=interface_kwargs,
        default_auto_reply="...",  # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
        skip_verify=False,  # note: you should set this to True if you expect your MemGPT AutoGen agent to call a function other than send_message on the first turn
        auto_save=False,  # Set this to True if you want the MemGPT AutoGen agent to save its internal state after each reply - you can also save manually with .save()
    )

    # If you'd like to save this agent at any point, you can do:
    # coder.save()

    # You can also autosave by setting auto_save=True, in which case coder.save() will be called automatically

    # To load an AutoGen+MemGPT agent you previously created, you can use the load function:
    #coder = load_autogen_memgpt_agent(agent_config={"name": "MemGPT_coder_v2"})


# Initialize the group chat between the user and two LLM agents (PM and coder)
groupchat = autogen.GroupChat(agents=[user_proxy, pm, coder], messages=[], max_round=12, speaker_selection_method="round_robin")
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Begin the group chat with a message from the user
user_proxy.initiate_chat(
    manager,
    message="I want to design an app to make me one million dollars in one month. Yes, your heard that right.",
)