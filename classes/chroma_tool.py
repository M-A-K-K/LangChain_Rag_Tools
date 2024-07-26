import os
import logging
from langchain import OpenAI, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.tools import BaseTool
from dotenv import load_dotenv

dotenv_path = os.path.join('utils', '.env')
load_dotenv(dotenv_path=dotenv_path)
openai_api_key = os.getenv('OPENAI_API_KEY')

# Retrieve API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')  # Replace with OpenAI API key
if not api_key:
    raise ValueError("OpenAI API key is not set in the environment variables.")

# conversation_memory = ConversationBufferWindowMemory(
#     memory_key='chat_history',
#     k=5,
#     return_messages=True
# )

class ChromaTool(BaseTool):
    name = "ChromaTool"
    description = """
    answer from the context.
    Use this tool when you have questions about blockchain and persons, people
    try to carefully search from the content.
    answer from the context.
    This is a retreiver tool which retreives information from content of data.
    Answer it generally if you do not have knowledge.
    """
    persist_directory = ""

    def __init__(self, persist_directory: str):
        super().__init__()  # Ensure that the parent class is initialized if needed
        self.persist_directory = persist_directory
    
    def _run(self, question: str):
        # Validate the directory
        if not os.path.exists(self.persist_directory):
            return {'error': f'Directory "{self.persist_directory}" does not exist'}
        
        # Print contents of the directory for debugging
        directory_contents = os.listdir(self.persist_directory)
        #logging.debug(f"Contents of '{self.persist_directory}': {directory_contents}")
        
        # Load FAISS index and set up Chroma vector store
        embedding = OpenAIEmbeddings(openai_api_key=api_key)
        vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=embedding)

        # Initialize the LLM
        llm = OpenAI(openai_api_key=api_key)

        # Define the custom prompt
        prompt_template = """
        Use this tool if information is about researchers from paper and about related
        Blockchain Based Scalable Domain Access.
        Answer from the context of documents.
        Try to find the answer from documents.
        Answer generally if tools do not have answer.
        {context}
        Question: {question}
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Set up the QA chain with conversational memory
        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_type="similarity_score_threshold", 
                                            search_kwargs={"score_threshold": 0.5}),
            chain_type_kwargs=chain_type_kwargs
        )

        # Run the QA chain to generate the response
        response = qa_chain(question)

        # Log conversation memory
        #logging.debug(f"Conversation memory: {conversation_memory.load_memory_variables({})}")
        if isinstance(response, dict):
            response = str(response)
        elif not isinstance(response, str):
            response = repr(response)  # Fallback to using repr for non-string objects

        return response
    
    def _arun(self, persist_directory: str, question: str):
        raise NotImplementedError("This tool does not support async operations.")
