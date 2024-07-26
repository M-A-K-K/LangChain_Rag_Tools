from flask import Flask, request, jsonify
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_openai import OpenAI
from langchain.agents import initialize_agent
import logging
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from classes.chroma_tool import ChromaTool
from classes.calculator_tool import AgeCalculatorTool
application = Flask(__name__)

# Load environment variables from the .env file
dotenv_path = os.path.join('utils', '.env')
load_dotenv(dotenv_path=dotenv_path)
openai_api_key = os.getenv('OPENAI_API_KEY')

# Retrieve API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')  # Replace with OpenAI API key
if not api_key:
    raise ValueError("OpenAI API key is not set in the environment variables.")
embedding = OpenAIEmbeddings(openai_api_key=api_key)
conversation_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)

@application.route('/create_embeddings', methods=['POST'])
def create_embeddings():
    data = request.json

    if 'pdf_path' not in data:
        return jsonify({'error': 'Missing PDF path in request body'}), 400

    pdf_path = data['pdf_path']

    if not os.path.exists(pdf_path):
        return jsonify({'error': 'The specified PDF file does not exist.'}), 400

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

         # Ensure this is correctly initialized
        db = Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")
        question = "Muhammad Usman"
        docs = db.similarity_search(question)

        # print results
        print(docs[0].page_content)
        return jsonify(docs[0].page_content), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

logging.basicConfig(level=logging.DEBUG)

# Initialize the memory object

@application.route('/generate_response', methods=['POST'])
def generate_response():
    try:
        data = request.json

        # Log incoming request
        logging.debug(f"Incoming request data: {data}")

        # Validate presence of required fields
        if 'question' not in data:
            return jsonify({'error': 'Missing "question" in request body'}), 400
        if 'persist_directory' not in data:
            return jsonify({'error': 'Missing "persist_directory" in request body'}), 400

        question = data['question']
        persist_directory = data['persist_directory']

        # Check if the persist_directory exists
        if not os.path.exists(persist_directory):
            return jsonify({'error': f'Directory "{persist_directory}" does not exist'}), 404
        
        # Print contents of the directory for debugging
        directory_contents = os.listdir(persist_directory)
        logging.debug(f"Contents of '{persist_directory}': {directory_contents}")

        # Load FAISS index and set up Chroma vector store
        
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

        # Initialize the LLM
        llm = OpenAI(openai_api_key=api_key)

        # Define the custom prompt
        prompt_template = """
        You are a helpful assistant.
        answer from context of documents.
        If you don't know answer,say I have no knowledge about it.
        {context}
        Question: {question}
        Chat History: {chat_history}
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

        # Set up the QA chain with conversational memory
        chain_type_kwargs = {"prompt": PROMPT, "memory": conversation_memory}
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
        logging.debug(f"Conversation memory: {conversation_memory.load_memory_variables({})}")

        return jsonify({'response': response, 'directory_contents': directory_contents, 'conversation_memory': conversation_memory.load_memory_variables({})})

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500


conversation_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

@application.route('/get_age', methods=['POST'])
def get_age():
    try:
        data = request.json
        birth_date = data.get('birth_date')
        
        # Validate presence of required fields
        if not birth_date:
            return jsonify({'error': 'Missing "birth_date" in request body'}), 400

        # Initialize LLM
        llm = OpenAI(api_key=api_key)

        # Define the custom prompt
        prompt_template = """Assistant is a large language model trained by OpenAI.
        Unfortunately, Assistant is terrible at age problems. When provided with math questions, no matter how simple, assistant always refers to its trusty tools and absolutely does NOT try to answer math questions by itself.
        {tools}
        Chat History: {history}
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["tools", "history"])

        # Initialize tools
        tools = [AgeCalculatorTool()]

        # Initialize agent with tools and prompt
        agent = initialize_agent(
            agent_type='chat-conversational-react-description',
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=conversation_memory,
            prompt=PROMPT,
            handle_parsing_errors=True
        )

        # Get the age using the agent
        result = agent.run(f"Calculate the age for birth date: {birth_date}")

        # Retrieve chat history from memory
        chat_history = conversation_memory.load_memory_variables({}).get('chat_history', [])

        # Convert chat history to a JSON-compatible format
        formatted_chat_history = []
        for message in chat_history:
            if hasattr(message, 'content'):
                formatted_chat_history.append({
                    'role': message.__class__.__name__,
                    'content': message.content
                })
            else:
                formatted_chat_history.append(str(message))  # Fallback to string conversion

        return jsonify({
            'response': result,
            'chat_history': formatted_chat_history
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@application.route('/process_question', methods=['POST'])
def process_question():
    try:
        data = request.json
        question = data.get('question')
        persist_directory = data.get('persist_directory')
        
        # Log received data
        logging.info(f"Received data: question='{question}', persist_directory='{persist_directory}'")
        
        # Validate presence of required fields
        if not question:
            return jsonify({'error': 'Missing "question" in request body'}), 400
        if not persist_directory:
            return jsonify({'error': 'Missing "persist_directory" in request body'}), 400

        # Instantiate and run the tool
        tool = ChromaTool(persist_directory='')
        result = tool._run(persist_directory=persist_directory, question=question)
        
        # Log result
        logging.info(f"Processing result: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500


conversation_history = []

def serialize_message(message):
    """Convert message objects to a JSON-serializable format."""
    if hasattr(message, 'text'):
        return {'text': message.text}
    return str(message)  # Fallback to  a string representation

@application.route('/process_request', methods=['POST'])
def process_request():
    global conversation_history
    
    try:
        data = request.json
        question = data.get('question')
        persist_directory = "./chroma_db"

        # logging.info(f"Received data: question='{question}', persist_directory='{persist_directory}'")

        if not question:
            return jsonify({'error': 'Missing "question" in request body'}), 400

        llm = ChatOpenAI(api_key=openai_api_key, model='gpt-3.5-turbo-16k')

        prompt_template = """You are an intelligent agent.
        Answer generally if tools do not have answer.
        use Chroma tool if information is about researchers from paper and about related
        Blockchain Based Scalable Domain Access
        use calculator_tool if user asks about age and age is in conversation history.
        Control Framework for Industrial Internet of Things.
        is user asks about previous responses return them their response.
        If users age is already calculate and he ask for his age again the use the chat history to respond the donot go for the Age Calculator Tool
        {tools}
        Chat History: {history}
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["tools", "history"])

        tools = [AgeCalculatorTool(), ChromaTool(persist_directory)]

        agent = initialize_agent(
            agent_type='chat-conversational-react-description',
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=conversation_memory,
            prompt=PROMPT,
            handle_parsing_errors=True
        )
        
        conversation_history.append({'user': question})

        result = agent.invoke(question)
        conversation_history.append({'agent': serialize_message(result)})
        logging.info(f"Received history: question='{conversation_memory.load_memory_variables({}).get('chat_history', [])}', persist_directory='{persist_directory}'")
        return jsonify({'result': serialize_message(result)})
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@application.route('/geqt', methods=['POST'])
def geqt():
  # setup the tools
  # 
  @tool
  def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

 
  @tool
  def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


  @tool
  def square(a) -> int:
    """Calculates the square of a number."""
    a = int(a)
    return a * a

  prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a mathematical assistant.
        Use your tools to answer questions. If you do not have a tool to
        answer the question, say so. 

        Return only the answers. e.g
        Human: What is 1 + 1?
        AI: 2
        """),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
  )

  # Choose the LLM that will drive the agent
  llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

  # setup the toolkit
  toolkit = [add, multiply, square]

  # Construct the OpenAI Tools agent
  agent = create_openai_tools_agent(llm, toolkit, prompt)

  # Create an agent executor by passing in the agent and tools
  agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

  result = agent_executor.invoke({"input": "what is 2 + 1?"})

  print(result['output'])
  return jsonify({'output': result['output']})


    
if __name__ == '__main__':
    application.run(debug=True)
