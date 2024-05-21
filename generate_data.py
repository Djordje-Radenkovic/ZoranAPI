###############################################################################
                     # Generate Embeddings for Target Repository
###############################################################################
                     
# import libraries
from dotenv import load_dotenv
import os
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import pickle

# load environment variables
load_dotenv()

# Access OpenAI API key
openai_key = os.getenv('OPENAI_API_KEY')

# Clone target repo locally
target_repo_link = "https://github.com/langchain-ai/langchain"
repo_path = "Users/djr/Desktop/target_repo"
repo = Repo.clone_from(target_repo_link, to_path=repo_path)

# Load repository content
loader = GenericLoader.from_filesystem(
    repo_path + "/libs/core/langchain_core",
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()

# index the data
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)

# generate and save vector embeddings 
db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()), persist_directory='vector_store')