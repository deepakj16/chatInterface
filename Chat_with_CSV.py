import streamlit as st
import pandas as pd
import os
import requests
import json
import openai
import inspect
import pyodbc
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

#Load environment variables
load_dotenv("credentials.env")

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
aoai_api_version = os.environ["AZURE_OPENAI_API_VERSION"] #"2023-05-15"
aoai_api_type = os.environ["AZURE_OPENAI_API_TYPE"]


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False
    if "CurrentPage" in st.session_state: 
        del st.session_state["CurrentPage"]
   
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

def main():
    st.set_page_config(page_title="Chat with CSV")    
    st.write(
    """
    # Chat with CSV! 
    Chat with CSV file. Upload CSV file and ask questions about the data, get answers from the AI model.

    """
    )

    if "CurrentPage" not in st.session_state or st.session_state["CurrentPage"] != "Chat with CSV":        
            #first time on this page: 
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
            st.session_state["CurrentPage"] = "Chat with CSV"

    uploaded_file = st.file_uploader(
        "Upload a Data file",
        type=list(file_formats.keys()),
        help="Supports CSV and Excel files!",
        on_change=clear_submit,
    )

    if uploaded_file:    
        c = load_data(uploaded_file)

        if "messages" not in st.session_state or st.button("Clear conversation history"):
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input(placeholder="What is this data about?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

        llm = AzureChatOpenAI(azure_endpoint=aoai_endpoint, 
                            openai_api_key = aoai_api_key, 
                            temperature=0, 
                            azure_deployment=deployment_name, 
                            openai_api_version=aoai_api_version, 
                            streaming=True)

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            number_of_head_rows=df.shape[0] #send all the rows to LLM
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

if __name__ == "__main__":
    main()
