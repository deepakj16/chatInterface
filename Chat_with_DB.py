import streamlit as st
import requests
import pandas as pd
import openai
import azure.cognitiveservices.speech as speechsdk
import pyodbc
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import time

st.set_page_config(layout="wide")


#Load environment variables
load_dotenv("credentials.env")

aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
aoai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
aoai_api_version_For_COSMOS = "2023-08-01-preview"        #### for cosmos other API version doesnt work. Hence this is not using env variable API version for cosmos
aoai_embedding_deployment = os.environ["AZURE_EMBEDDING_MODEL"]


sql_server_name = os.environ["SQL_SERVER_NAME"]
sql_server_db = os.environ["SQL_SERVER_DATABASE"]
sql_server_username = os.environ["SQL_SERVER_USERNAME"]
sql_Server_pwd = os.environ["SQL_SERVER_PASSWORD"]
SQL_ODBC_DRIVER_PATH = os.environ["SQL_ODBC_DRIVER_PATH"]

COSMOS_MONGO_CONNECTIONSTRING = os.environ["COSMOS_MONGO_CONNECTIONSTRING"] 
COSMOS_MONGO_DBNAME = os.environ["COSMOS_MONGO_DBNAME"]
COSMOS_MONGO_CONTAINER = os.environ["COSMOS_MONGO_CONTAINER"]
COSMOS_MONGO_API = os.environ["COSMOS_MONGO_API"]

collection_name = COSMOS_MONGO_CONTAINER
# Vector search index parameters
index_name = "VectorSearchIndex"
vector_dimensions = (
    1536  # text-embedding-ada-002 uses a 1536-dimensional embedding vector
)
num_lists = 1
similarity = "COS"  # cosine distance

def connect_sql_server():
    '''
    Setup SQL Server
    '''
    conn = pyodbc.connect('DRIVER={'+ SQL_ODBC_DRIVER_PATH +'};SERVER='+sql_server_name+';DATABASE='+sql_server_db+';UID='+sql_server_username+';PWD='+ sql_Server_pwd)
    cursor = conn.cursor()
    
    return conn


def run_sql_query(aoai_sqlquery):
    '''
    Function to run the generated SQL Query on SQL server and retrieve output.
    Input: AOAI completion (SQL Query)
    Output: Pandas dataframe containing results of the query run    
    '''
    conn = connect_sql_server()
    df = pd.read_sql(aoai_sqlquery, conn)
    return df

def generate_nl_to_sql(userPrompt):
    '''
    This GPT4 engine is setup for NLtoSQL tasks on the Sales DB.
    Input: NL question related to BikeStores sales
    Output: SQL query to run on the db1 database and BikeStores schema
    '''

    messages=[
            {"role": "system", "content": """ You are a SQL programmer Assistant.Your role is to generate SQL code (SQL Server) to retrieve an answer to a natural language query. Make sure to disambiguate column names when creating queries that use more than one table. If a valid SQL query cannot be generated, only say "ERROR:" followed by why it cannot be generated.
                  Do not answer any questions on inserting or deleting rows from the table. Instead, say "ERROR: I am not authorized to make changes to the data"
                  Dont generate SQL queries startign with ```sql or starting with single or double quotes.

                  Use the following BikeStores database schema to write SQL queries:
                  BikeStores.customers(customer_id INTEGER, first_name VARCHAR, last_name VARCHAR, email VARCHAR, phone VARCHAR, street VARCHAR, city VARCHAR, state VARCHAR, zip_code VARCHAR, PRIMARY KEY (customer_id))
                  BikeStores.products(product_id INTEGER,product_name varchar, list_price DECIMAL(10,2), category_id INTEGER, model_year INTEGER,brand_id INTEGER, PRIMARY KEY(product_id), FOREIGN KEY(category_id, brand_id))
                  BikeStores.stocks(product_id INTEGER, store_id INTEGER, quantity INTEGER, PRIMARY KEY(store_id, product_id), FOREIGN KEY(store_id, product_id))
                  BikeStores.categories(category_id INTEGER, category_name VARCHAR, PRIMARY KEY(category_id))
                  BikeStores.brands(brand_id INTEGER, brand_name VARCHAR, PRIMARY KEY(brand_id))
                  BikeStores.stores(store_id INTEGER, store_name VARCHAR, PRIMARY KEY(store_id))                  

                  Examples:
                  User: List all Bicycle products, along with their prices. SQL Code:
                  Assistant: SELECT [product_id],[product_name] ,[list_price] FROM [BikeStores].[products] where product_name like '%Bicycle%';
                  User: Which is the cheapest product ? SQL Code:
                  Assistant: SELECT TOP 1 product_name, list_price FROM BikeStores.products ORDER BY list_price ASC;
                  User: List all products in BikeStores with list price which belongs to "Mountain Bikes" category?
                  Assistant: select p.product_name, p.list_price from [BikeStores].[products] p inner join BikeStores.categories as pc on p.category_id = pc.category_id where pc.category_name = 'Mountain Bikes';
                  User: List 10 products from BikeStores alongwith their prices for Electra brand ?
                  Assistant: select TOP 10 p.product_name, p.list_price from [BikeStores].[products] p inner join BikeStores.brands as b on p.brand_id = b.brand_id where b.brand_name = 'Electra';
                  User: List customers in New York city ?
                  Assistant: SELECT [customer_id] ,[first_name],[last_name] FROM [BikeStores].[customers]  where city = 'New York';
            """}
        ]

    messages.extend(userPrompt)    
    
    client = openai.AzureOpenAI(
        base_url=f"{aoai_endpoint}/openai/deployments/{deployment_name}/",        
        api_key=aoai_api_key,
        api_version="2023-12-01-preview"
    )
    
    response = client.chat.completions.create(
        model=deployment_name, # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.

        messages=messages,                 
        temperature=0,
        max_tokens=2000
             
    )
    return response
    
def handle_chat_SQLDB(prompt):
    # Echo the user's prompt to the chat window
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send the user's prompt to Azure OpenAI and display the response
    # The call to Azure OpenAI is handled in create_chat_completion()
    # This function loops through the responses and displays them as they come in.
    # It also appends the full response to the chat history.

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = generate_nl_to_sql(st.session_state.messages)        
        response_message = response.choices[0].message
        full_response += ("" + response_message.content + "\n" or "")        
        message_placeholder.markdown(full_response)
        full_response += ("\n SQL Output: \n" or "")
        st.markdown(full_response)
        sql_output = run_sql_query(response_message.content)
        full_response += (sql_output.to_string() or "")
                   
        message_placeholder.markdown(st.dataframe(sql_output))
    st.session_state.messages.append({"role": "assistant", "content": full_response})
      
def handle_chat_sql_langchain(prompt, db_chain):
    return db_chain(prompt)['result']


def handle_chat_cosmos(prompt):
    # Echo the user's prompt to the chat window
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send the user's prompt to Azure OpenAI and display the response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response_message = "" #response.choices[0].message
        full_response += ("" + response_message.content + "\n" or "")        
        message_placeholder.markdown(full_response)        
        st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

### 02: Chat with customer data
def create_chat_cosmos(messages):
    # Create an Azure OpenAI client. We create it in here because each exercise will
    # require at a minimum different base URLs.

    client = openai.AzureOpenAI(        
        base_url=f"{aoai_endpoint}/openai/deployments/{deployment_name}/extensions/",
        api_key=aoai_api_key,
        api_version=aoai_api_version_For_COSMOS
    )
    
    # Create and return a new chat completion request
    # Be sure to include the "extra_body" parameter to use Cosmos as the data source
    #this is Azure Open AI On your data feature

    return client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ],
        stream=True,
        extra_body={
            "dataSources": [
                {
                    "type": "AzureCosmosDB",
                    "parameters": {
                        "connectionString": COSMOS_MONGO_CONNECTIONSTRING,
                        "indexName": index_name,
                        "containerName": COSMOS_MONGO_CONTAINER,
                        "databaseName": COSMOS_MONGO_DBNAME,
                        "fieldsMapping": {
                            "contentFieldsSeparator": "\n",
                            "contentFields": ["text"],
                            "filepathField": "id",
                            "titleField": "description",
                            "urlField": None,
                            "vectorFields": ["embedding"],
                        },
                        "inScope": "true",
                        "roleInformation": "You are an AI assistant that helps people find information from retrieved data",
                        "embeddingEndpoint": f"{aoai_endpoint}/openai/deployments/{aoai_embedding_deployment}/embeddings/",
                        "embeddingKey": aoai_api_key,
                        "strictness": 3,
                        "topNDocuments": 5,
                    }
                }
            ]
        }
    )


def handle_chat_cosmos(prompt):
    # Echo the user's prompt to the chat window
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send the user's prompt to Azure OpenAI and display the response
    # The call to Azure OpenAI is handled in create_chat_completion()
    # This function loops through the responses and displays them as they come in.
    # It also appends the full response to the chat history.

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in create_chat_cosmos(st.session_state.messages):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

def handle_prompt(chat_option, prompt):
    if chat_option == "SQL DB":
        handle_chat_SQLDB(prompt)
   
    elif chat_option == "Cosmos DB":
        handle_chat_cosmos(prompt)        
    else:
        st.write("Please select a chat option before calling the chatbot.")

def option_changed():
    if "CurrentPage" in st.session_state: 
        del st.session_state["CurrentPage"]
    # Delete all the items in Session state
    # for key in st.session_state.keys():
    #     del st.session_state[key]

def main():
    #st.session_state.messages = []
    st.write(
    """
    # Chat with Database! 
    This proof of concept is intended to serve as a demonstration of Azure OpenAI's capabilities to chat over databases like SQL DB, Cosmos DB etc.
    \n Use Case : A local bike store has a SQL database with information about their customers, products, and stores. This is simple chat application which allows users to ask questions in natural language
     and smart AI application queries the database and get answers in the form of SQL queries that are executed to get final response from LLM.

    """
    )
    tooltip_text = """SQL DB Examples:  \nList all Bicycle products, along with their prices;  \nWhich is the cheapest product  \nList 10 products from BikeStores alongwith their prices for Electra brand  \nList customers in New York city?  \nList all products in BikeStores with list price and category
                      \n  \n Cosmos DB Examples:  \nGive me details about 2017 Trek Fuel EX 5 27.5 Plus  \nWhich bike is better for mountain biking  \nlist few high-performance electric bike along with their list price  \nWhat is list price of Surly Straggler bike  \nwhich bike is good option for kids  \nwhich road bikes are engineered for speed and comfort both? 

                   """
    chat_option = st.radio(label="Choose the chat option you want to try:", options=["SQL DB", "Cosmos DB"], help=tooltip_text, on_change=option_changed)

    if "CurrentPage" not in st.session_state or st.session_state["CurrentPage"] != "Chat with DB":        
        #first time on this page: 
        st.session_state["messages"] = []
        st.session_state["CurrentPage"] = "Chat with DB"
   
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Await a user message and handle the chat prompt when it comes in.
    if prompt := st.chat_input("Enter a message:"):
        handle_prompt(chat_option, prompt)

if __name__ == "__main__":
    main()



