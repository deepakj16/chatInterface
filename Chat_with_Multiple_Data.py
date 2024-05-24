import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler

import os
import re
import asyncio
import random
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import BingSearchAPIWrapper
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.runnables import ConfigurableField, ConfigurableFieldSpec
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory, CosmosDBChatMessageHistory
from langchain.agents import ConversationalChatAgent, AgentExecutor, Tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain_core.runnables.history import RunnableWithMessageHistory
from datetime import datetime
from dotenv import load_dotenv

#custom libraries that we will use later in the app
from common.utils import (
    DocSearchAgent, 
    CSVTabularAgent, 
    SQLSearchAgent, 
    ChatGPTTool, 
    BingSearchAgent
)
from common.prompts import CUSTOM_CHATBOT_PROMPT
from IPython.display import Markdown, HTML, display 

st.set_page_config(layout="wide")

#Load environment variables
load_dotenv("credentials.env")

aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
aoai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]

sql_server_name = os.environ["SQL_SERVER_NAME"]
sql_server_db = os.environ["SQL_SERVER_DATABASE"]
sql_server_username = os.environ["SQL_SERVER_USERNAME"]
sql_Server_pwd = os.environ["SQL_SERVER_PASSWORD"]

blob_connection_string = os.environ["BLOB_CONNECTION_STRING"]
blob_sas_token = os.environ["BLOB_SAS_TOKEN"]

azure_cosmos_endpoint = os.environ["AZURE_COSMOSDB_ENDPOINT"]
azure_cosmos_db = os.environ["AZURE_COSMOSDB_NAME"]
azure_cosmos_container = os.environ["AZURE_COSMOSDB_CONTAINER"]
azure_cosmos_connection = os.environ["AZURE_COSMOSDB_CONNECTION_STRING"]

azure_search_covid_index = os.environ["AZURE_SEARCH_COVID_INDEX"]
bing_Search_url = os.environ["BING_SEARCH_URL"]
bing_subscription_key = os.environ["BING_SUBSCRIPTION_KEY"]
cog_services_name = os.environ["COG_SERVICES_NAME"]
cog_services_key = os.environ["COG_SERVICES_KEY"]

def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False

def get_session_history(session_id: str, user_id: str) -> CosmosDBChatMessageHistory:
    cosmos = CosmosDBChatMessageHistory(
        cosmos_endpoint=azure_cosmos_endpoint,
        cosmos_database=azure_cosmos_db,
        cosmos_container=azure_cosmos_container,
        connection_string=azure_cosmos_connection,
        session_id=session_id,
        user_id=user_id
        )

    # prepare the cosmosdb instance
    cosmos.prepare_cosmos()
    return cosmos
   

def handle_chat_langchain(prompt, brain_agent_executor, config, cb_handler):    
    with st.chat_message("user"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response =  brain_agent_executor.invoke({"question": prompt},config=config)["output"]
        full_response += response
        clear_submit()
        #because of open bug in streamlit callback handler - we need to explicitly mark this as completed
        #https://github.com/streamlit/streamlit/pull/8311
        cb_handler._current_thought._container.update(
                                                        label="Completed",
                                                        state="complete",
                                                        expanded=False,
                                                    )
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    return response

def handle_chat_semantickernel(prompt):
    return "To be Implemented.."

def option_changed():
    if "CurrentPage" in st.session_state: 
        del st.session_state["CurrentPage"]    

def main():
    st.write(
    """
    # Chat with Multiple Data Sources! 
    Chat with Multiple Data sources using Orchestration Frameworks like Langchain, semantic kernel etc.
    \n <b>Use Case</b>: Covid data is scattered across various data sources. This chat based application can pull data from multiple data sources 
    for example - unstructured data sources (PDF, word, Image files);  structured data sources (SQL database) ; Bing search engine and GPT - the Large Language Model iteself. 
    """, unsafe_allow_html=True
    )
    col1, col2, col3, col4 = st.columns(4)    
    with col3:
        loggedInUser = st.selectbox("Change Login:", ("Guest", "Jennifer", "Steve"), index=0, placeholder="Select user...")       

    with col2:
        st.write ("Logged in as: <span style='color:blue'>", loggedInUser, "</span>", unsafe_allow_html=True)    
    tooltip_text = """Examples:  \n @docsearch how many covid cases globally?  \n @docsearch Explain Variants of SARS-Cov-2
    \n @sql how many people died during covid?  \n @sql how many confirmed deaths occurred in Colorado due to covid. 
    \n @bing what is the latest news on covid?  \n @chatgpt what are symptoms of Covid """
    with col1:
        chat_option = st.radio(label="Choose the chat option you want to try:", options=["Langchain", "Semantic Kernel"], help=tooltip_text, on_change=option_changed)
    
    if "CurrentPage" not in st.session_state or st.session_state["CurrentPage"] != "Chat with Multiple Data":        
        #first time on this page: 
        st.session_state["messages"] = []
        st.session_state["CurrentPage"] = "Chat with Multiple Data"
    else:
        #do nothing - this is second or more time on this page
        a= None

    st.divider()  
   
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Extract info from TurnContext - You can change this to whatever , this is just one option 
    
    user_id = loggedInUser 
    session_id = user_id + '-' + f'{datetime.now():%Y%m%d-%H}'

    if chat_option == "Langchain":
        # Set Callback Handler
        cb_handler = StreamlitCallbackHandler(st.container(),  expand_new_thoughts=True, collapse_completed_thoughts = True)
        #cb_handler = BotServiceCallbackHandler(turn_context)
        cb_manager = CallbackManager(handlers=[cb_handler])


        # Set LLM 
        llm = AzureChatOpenAI(azure_endpoint=aoai_endpoint, 
                                openai_api_key = aoai_api_key,                            
                                azure_deployment=deployment_name, 
                                openai_api_version=aoai_api_version, 
                                streaming=False, 
                                temperature=0.5,
                                max_tokens=1000 
                                ,callback_manager=cb_manager
                            )

        # Initialize our Tools/Experts
        text_indexes = [azure_search_covid_index] #, "cogsrch-index-csv"]
        doc_search = DocSearchAgent(llm=llm, indexes=text_indexes,
                            k=10, similarity_k=4, reranker_th=1,
                            sas_token=blob_sas_token, name="docsearch",
                            verbose=False) #callback_manager=cb_manager
     
        www_search = BingSearchAgent(llm=llm, k=5,verbose=False)
        sql_search = SQLSearchAgent(llm=llm, k=10,verbose=False)
        chatgpt_search = ChatGPTTool(llm=llm)     #callback_manager=cb_manager
        #csv_search = CSVTabularTool(path="src\data\BikeStores_Staff.csv", llm=llm, callback_manager=cb_manager, return_direct=True)
        tools = [www_search, sql_search, doc_search, chatgpt_search] #csv_search,  book_search]

        agent = create_openai_tools_agent(llm, tools, CUSTOM_CHATBOT_PROMPT)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        brain_agent_executor = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
        )
                
        config={"configurable": {"session_id": session_id, "user_id": user_id}}  

    # Await a user message and handle the chat prompt when it comes in.
    if prompt := st.chat_input("Enter a message:"):
        if chat_option == "Langchain":            
            handle_chat_langchain(prompt, brain_agent_executor, config, cb_handler)
        elif chat_option == "Semantic Kernel":
            st.write("<h6 style='text-align: center; color: red;'>This feature is not yet implemented.</h5>", unsafe_allow_html=True)
            #handle_chat_semantickernel(prompt)    
        else:
            st.write("Please select a chat option before calling the chatbot.")        

if __name__ == "__main__":
    main()

