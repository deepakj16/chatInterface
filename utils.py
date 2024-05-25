import re
import os
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Awaitable, Callable, Tuple, Type, Union
import requests
import asyncio

from collections import OrderedDict
import base64
from bs4 import BeautifulSoup
import docx2txt
import tiktoken
import html
import time
from time import sleep
from typing import List, Tuple
from pypdf import PdfReader, PdfWriter
from dataclasses import dataclass
from sqlalchemy.engine.url import URL
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor, initialize_agent, AgentType, Tool
from langchain_community.utilities import BingSearchAPIWrapper
from langchain.agents import create_sql_agent, create_openai_tools_agent
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.callbacks.base import BaseCallbackManager
from langchain.requests import RequestsWrapper
from langchain.chains import APIChain
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.utils.json_schema import dereference_refs
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from operator import itemgetter
from typing import List



try:
    from .prompts import (AGENT_DOCSEARCH_PROMPT, CSV_PROMPT_PREFIX, MSSQL_AGENT_PREFIX, MSSQL_AGENT_SUFFIX,
                          MSSQL_AGENT_FORMAT_INSTRUCTIONS, CHATGPT_PROMPT, BINGSEARCH_PROMPT, APISEARCH_PROMPT,DOCSEARCH_PROMPT)
except Exception as e:
    print(e)
    from prompts import (AGENT_DOCSEARCH_PROMPT, CSV_PROMPT_PREFIX, MSSQL_AGENT_PREFIX, MSSQL_AGENT_SUFFIX,
                          MSSQL_AGENT_FORMAT_INSTRUCTIONS, CHATGPT_PROMPT, BINGSEARCH_PROMPT, APISEARCH_PROMPT)




######## AGENTS AND TOOL CLASSES #####################################
###########################################################
    
class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")

class GetDocSearchResults_Tool(BaseTool):
    name = "docsearch"
    description = "useful when the questions includes the term: docsearch"
    args_schema: Type[BaseModel] = SearchInput
    
    indexes: List[str] = []
    k: int = 10
    reranker_th: int = 1
    sas_token: str = "" 

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:

        retriever = CustomAzureSearchRetriever(indexes=self.indexes, topK=self.k, reranker_threshold=self.reranker_th, 
                                               sas_token=self.sas_token, callback_manager=self.callbacks)
        results = retriever.get_relevant_documents(query=query)
        
        return results

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        
        retriever = CustomAzureSearchRetriever(indexes=self.indexes, topK=self.k, reranker_threshold=self.reranker_th, 
                                               sas_token=self.sas_token, callback_manager=self.callbacks)
        # Please note below that running a non-async function like run_agent in a separate thread won't make it truly asynchronous. 
        # It allows the function to be called without blocking the event loop, but it may still have synchronous behavior internally.
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(ThreadPoolExecutor(), retriever.get_relevant_documents, query)
        
        return results


class DocSearchAgent(BaseTool):
    """Agent to interact with for Azure AI Search """
    
    name = "docsearch"
    description = "useful when the questions includes the term: docsearch.\n"
    args_schema: Type[BaseModel] = SearchInput

    llm: AzureChatOpenAI
    indexes: List[str] = []
    k: int = 10
    reranker_th: int = 1
    sas_token: str = ""   
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            tools = [GetDocSearchResults_Tool(indexes=self.indexes, k=self.k, reranker_th=self.reranker_th, sas_token=self.sas_token)]

            agent = create_openai_tools_agent(self.llm, tools, AGENT_DOCSEARCH_PROMPT)

            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, callback_manager=self.callbacks, handle_parsing_errors=True)

            response = agent_executor.invoke({"question":query})['output']
            
            return response

        except Exception as e:
            print(e)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("DocSearchTool does not support async")
    
    

class CSVTabularAgent(BaseTool):
    """Agent to interact with CSV files"""
    
    name = "csvfile"
    description = "useful when the questions includes the term: csvfile.\n"
    args_schema: Type[BaseModel] = SearchInput

    path: str
    llm: AzureChatOpenAI
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        
        # try:
        agent_executor = create_csv_agent(self.llm, self.path, verbose=self.verbose, 
                                 callback_manager=self.callbacks,
                                 agent_type="openai-tools",
                                 prefix=CSV_PROMPT_PREFIX)

        response = agent_executor.invoke(query)['output'] 


        return response
        # except Exception as e:
        #     print(e)
        #     response = e
        #     return response
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CSVTabularTool does not support async")
        
        
class SQLSearchAgent(BaseTool):
    """Agent to interact with SQL databases"""
    
    name = "sqlsearch"
    description = "useful when the questions includes the term: sqlsearch.\n"
    args_schema: Type[BaseModel] = SearchInput

    llm: AzureChatOpenAI
    k: int = 30
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        db_config = {
            'drivername': 'mssql+pyodbc',
            'username': os.environ["SQL_SERVER_USERNAME"] +'@'+ os.environ["SQL_SERVER_NAME"],
            'password': os.environ["SQL_SERVER_PASSWORD"],
            'host': os.environ["SQL_SERVER_NAME"],
            'port': 1433,
            'database': os.environ["SQL_SERVER_DATABASE"],
            'query': {'driver': 'ODBC Driver 17 for SQL Server'}
        }

        db_url = URL.create(**db_config)
        db = SQLDatabase.from_uri(db_url)
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        agent_executor = create_sql_agent(
            prefix=MSSQL_AGENT_PREFIX,
            suffix=MSSQL_AGENT_SUFFIX,
            format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS,
            llm=self.llm,
            toolkit=toolkit,
            callback_manager=self.callbacks,
            top_k=self.k,
            verbose=self.verbose,
            agent_type="openai-tools"
        )

        for i in range(2):
            try:
                response = agent_executor.invoke(query)["output"] 
                break
            except Exception as e:
                response = str(e)
                continue

        return response
        
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SQLDbTool does not support async")
        
        
        
class ChatGPTTool(BaseTool):
    """Tool for a ChatGPT clone"""
    
    name = "chatgpt"
    description = "default tool for general questions, profile or greeting like questions.\n"
    args_schema: Type[BaseModel] = SearchInput

    llm: AzureChatOpenAI
    
    def _run(self, query: str) -> str:
        try:
            chatgpt_chain = LLMChain(
                llm=self.llm, 
                prompt=CHATGPT_PROMPT,
                callback_manager=self.callbacks,
                verbose=self.verbose
            )

            response = chatgpt_chain.invoke(query)["text"]

            return response
        except Exception as e:
            print(e)
            
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("ChatGPTTool does not support async")
        
    
    
class GetBingSearchResults_Tool(BaseTool):
    """Tool for a Bing Search Wrapper"""

    name = "Searcher"
    description = "useful to search the internet.\n"
    args_schema: Type[BaseModel] = SearchInput

    k: int = 5
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        bing = BingSearchAPIWrapper(k=self.k)
        try:
            return bing.results(query,num_results=self.k)
        except:
            return "No Results Found"
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        bing = BingSearchAPIWrapper(k=self.k)
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(ThreadPoolExecutor(), bing.results, query, self.k)
            return results
        except:
            return "No Results Found"
            

class BingSearchAgent(BaseTool):
    """Agent to interact with Bing"""
    
    name = "bing"
    description = "useful when the questions includes the term: bing.\n"
    args_schema: Type[BaseModel] = SearchInput
    
    llm: AzureChatOpenAI
    k: int = 5
    
    def parse_html(self, content) -> str:
        soup = BeautifulSoup(content, 'html.parser')
        text_content_with_links = soup.get_text()
        return text_content_with_links

    def fetch_web_page(self, url: str) -> str:
        HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'}
        response = requests.get(url, headers=HEADERS)
        return self.parse_html(response.content)


    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            
            web_fetch_tool = Tool.from_function(
                func=self.fetch_web_page,
                name="WebFetcher",
                description="useful to fetch the content of a url"
            )

            tools = [GetBingSearchResults_Tool(k=self.k), web_fetch_tool]
            agent = create_openai_tools_agent(self.llm, tools, BINGSEARCH_PROMPT)

            agent_executor = AgentExecutor(agent=agent, tools=tools,
                                            return_intermediate_steps=True,
                                            callback_manager=self.callbacks,
                                            verbose=self.verbose,
                                            handle_parsing_errors=True)

            parsed_input = self._parse_input(query)
            response = agent_executor.invoke({"question":parsed_input})['output']

            return response

        except Exception as e:
            print(e)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchTool does not support async")
        
        
        
class GetAPISearchResults_Tool(BaseTool):
    """APIChain as a tool"""
    
    name = "apisearch"
    description = "useful when the questions includes the term: apisearch.\n"
    args_schema: Type[BaseModel] = SearchInput

    llm: AzureChatOpenAI
    api_spec: str
    headers: dict = {}
    limit_to_domains: list = []
    verbose: bool = False
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        
        chain = APIChain.from_llm_and_api_docs(
                            llm=self.llm,
                            api_docs=self.api_spec,
                            headers=self.headers,
                            verbose=self.verbose,
                            limit_to_domains=self.limit_to_domains
                            )
        try:
            sleep(2) # This is optional to avoid possible TPM rate limits
            response = chain.invoke(query)
        except Exception as e:
            response = e
        
        return response
            
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This Tool does not support async")
        

class APISearchAgent(BaseTool):
    """Agent to interact with any API given a OpenAPI 3.0 spec"""
    
    name = "apisearch"
    description = "useful when the questions includes the term: apisearch.\n"
    args_schema: Type[BaseModel] = SearchInput
    
    llm: AzureChatOpenAI
    llm_search: AzureChatOpenAI
    api_spec: str
    headers: dict = {}
    limit_to_domains: list = None
    verbose: bool = False
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            tools = [GetAPISearchResults_Tool(llm=self.llm,
                                              llm_search=self.llm_search,
                                              api_spec=str(self.api_spec),
                                              headers=self.headers,
                                              verbose=self.verbose,
                                              limit_to_domains=self.limit_to_domains)]
            
            parsed_input = self._parse_input(query)
            
            agent = create_openai_tools_agent(llm=self.llm, tools=tools, prompt=APISEARCH_PROMPT)
            agent_executor = AgentExecutor(agent=agent, tools=tools, 
                                           verbose=self.verbose, 
                                           return_intermediate_steps=True,
                                           callback_manager=self.callbacks)

            
            for i in range(2):
                try:
                    response = agent_executor.invoke({"question":parsed_input})["output"]
                    break
                except Exception as e:
                    response = str(e)
                    continue

            return response
        
        except Exception as e:
            print(e)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("APISearchAgent does not support async")
        
