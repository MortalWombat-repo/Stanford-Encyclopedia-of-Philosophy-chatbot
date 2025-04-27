import pysqlite3  # noqa: F401
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pickle
import time
import requests
import zipfile
from google import genai
from google.genai import types
from IPython.display import Markdown
from IPython.display import display
from dotenv import load_dotenv
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
import re

def import_google_api():
    #importing Google api key
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    for m in client.models.list():
        if "embedContent" in m.supported_actions:
            print(m.name)

    return client

def embedding_function(client):
    class GeminiEmbeddingFunction(EmbeddingFunction):
        document_mode = True

        def __init__(self, client):
            self.client = client
            self._retry = retry.Retry(predicate=lambda e: isinstance(e, genai.errors.APIError) and e.code in {429, 503})

        def __call__(self, input: Documents) -> Embeddings:
            embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
            response = self._retry(self.client.models.embed_content)(
                model="models/text-embedding-004",
                contents=input,
                config=types.EmbedContentConfig(task_type=embedding_task),
            )
            return [e.values for e in response.embeddings]
    
    return GeminiEmbeddingFunction(client)

def unpickle():
    #unpickling
    file_path = './pickled_dictionary.pkl'
    #variable_name = 'pickled_dictionary'

    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    pickled_dictionary = loaded_data

    # embedding URL to a list
    concatenated_list = [f"URL is {url} and the text is: {text}" for url, text in pickled_dictionary.items()]
    # "URL is https://plato.stanford.edu/entries/abduction/ and the text is: ... (extracted text) ..."
    return pickled_dictionary, concatenated_list

def create_collection(chroma_client, gemini_embedding_function, concatenated_list):
    # Get or create the collection
    DB_NAME = "googlecardb"
    embed_fn = gemini_embedding_function
    embed_fn.document_mode = True
    
    db = chroma_client.get_or_create_collection(
        name=DB_NAME,
        metadata={"model": "text-embedding-004", "dimension": 768},
        embedding_function=embed_fn
    )

    # Add documents to the collection
    documents = concatenated_list
    for i, doc in enumerate(documents):
        db.add(documents=[doc], ids=[str(i)])
        time.sleep(0.5)  # Delay to avoid rate limits
        print(f"Added document with ID: {i}, Content (first 100 chars): {str(doc[:100])}")

def unzipping_the_dataset():
    # URL of the zip file
    url = "https://huggingface.co/datasets/Icosar/chromadb_for_SEP_chatbot/resolve/main/output.zip"
    local_zip_path = "output.zip"
    extract_path = "./output"

    # Check if the output folder already has files
    if os.path.exists(extract_path) and os.listdir(extract_path):
        print("Output folder already exists and is not empty. Skipping download and extraction.")
    else:
        # Ensure the output directory exists
        os.makedirs(extract_path, exist_ok=True)

        # Download the zip file
        print("Downloading zip file...")
        response = requests.get(url)
        with open(local_zip_path, 'wb') as f:
            f.write(response.content)
        print("Download complete!")

        # Unzip the file
        print("Unzipping...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Unzipping complete!")

        # Delete the zip file
        os.remove(local_zip_path)
        print("Zip file removed. Done!")

def persistent_client(embed_fn):
    unzipping_the_dataset()
    
    # Initialize PersistentClient with desired path
    persist_dir = "./output"  # Use one directory for persistence
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    
    DB_NAME = "googlecardb"
    embed_fn = embed_fn
    collection = chroma_client.get_collection(DB_NAME, embedding_function=embed_fn)

    # List collection names to verify the database
    #collection_names = chroma_client.list_collections()
    #print("Collection names:", collection_names)

    # Access a specific collection by its name
    #collection_name = collection_names[0]  # You can select the first collection or any other collection by name

    # Peek into the collection (view first item)
    #print(collection.peek(1))
    #print(collection.count())
    print(collection.metadata)  # Check metadata for clues about the embedding model
    #print(collection.count())  # Verify the collection has data

    # Peek at a sample document
    #print(f"Sample document: {collection.peek(1)}")

    return embed_fn, collection

def get_article(user_query, embed_fn, collection, client):
    # Switch to query mode when generating embeddings
    embed_fn.document_mode = False

    result = collection.query(query_texts=[user_query], n_results=1)
    [all_passages] = result["documents"]

    query_oneline = user_query.replace("\n", " ")

    print(query_oneline)

    prompt = f"""
    You are a helpful and informative bot answering questions based on the reference passage provided below.

    Your style:
    If there is a url present in the text, add it at the start and separate with a new line.
    The text to look out for looks like this:
    "URL is [url to display]"
    Prioritize the display of the URL of of a person if finding conflicting information.

    For example "Descartes" and "Descartes epistemology" choose the URL of Descartes.
    Respond with the knowledge and clarity of a philosophy postdoc explaining complex ideas to a non-specialist audience.
    Do not start with phrases like "Sure, I can help you with that!". Directly begin with the relevant information.
    Start the answer with: "Here is some information about..."
    Break down complicated concepts clearly.

    Answering instructions:
    Respond in complete, comprehensive sentences (minimum 100 words).
    Include relevant background information.
    If the passage is irrelevant, you may answer based on general knowledge.

    Formatting instructions:
    If asked about a person, state their full name first (if available).
    Bold the name of any philosopher mentioned.
    If listing accomplishments, format them as a bullet-point list.
    If the philosopher founded a school of thought:
    Outline its main points clearly.
    List all known students and briefly describe their philosophies.
    If the philosopher had notable opponents:
    - Summarize their contrasting views.
    - Compare the philosopher's views and the opponents' views clearly.
    - **Always create a comparison table summarizing these differences.**
    - **The table must appear at the end of the answer, even if the contrasts are few or subtle.**

    MANDATORY:
    If there is a url present in the text, add it at the top.
    The text to look out for looks like this:
    "URL is [url to display]"
    Prioritize the display of the URL of of a person if finding conflicting information.
    For example "Descartes" and "Descartes epistemology" choose the URL of Descartes.

    QUESTION: {query_oneline}
    """

    for passage in all_passages:
        passage_oneline = passage.replace("\n", " ")
        prompt += f"PASSAGE: {passage_oneline}\n"

    answer = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt)

    #return Markdown(answer.text)
    return answer.text

def summarize_article(user_query, embed_fn, collection, client):
    # Switch to query mode when generating embeddings
    embed_fn.document_mode = False

    result = collection.query(query_texts=[user_query], n_results=1)
    [all_passages] = result["documents"]

    query_oneline = user_query.replace("\n", " ")

    prompt = f"""
    You are a helpful and informative bot answering questions based on the reference passage provided below.

    Your style:
    Respond with the knowledge and clarity of a philosophy postdoc explaining complex ideas to a non-specialist audience.
    Do not start with phrases like "Sure, I can help you with that!". Directly begin with the relevant information.
    Start the answer with: "Here is some information about..."
    Break down complicated concepts clearly.

    Answering instructions:
    Respond in bullet points separated by headings in the style of table of contents.
    Include relevant background information.
    If the passage is irrelevant, you may answer based on general knowledge.

    Formatting instructions:
    If asked about a person, state their full name first (if available).
    Bold the name of any philosopher mentioned.
    If listing accomplishments, format them as a bullet-point list.
    If the philosopher founded a school of thought:
    Outline its main points clearly.
    List all known students and briefly describe their philosophies.
    If the philosopher had notable opponents:
    - Summarize their contrasting views.
    - Compare the philosopher's views and the opponents' views clearly.

    QUESTION: {query_oneline}
    """

    for passage in all_passages:
        passage_oneline = passage.replace("\n", " ")
        prompt += f"PASSAGE: {passage_oneline}\n"

    answer = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt)

    #return Markdown(answer.text)
    return answer.text

def get_full_article(user_query, embed_fn, collection, client):

    # Switch to query mode when generating embeddings.
    embed_fn.document_mode = False

    result = collection.query(query_texts=[user_query], n_results=1)
    [all_passages] = result["documents"]

    #Markdown(all_passages[0])
    # Assuming all_passages[0] is the text containing "Bibliography"
    text = all_passages[0]
    #print(text)

    # Regular expression to delete everything after the second occurrence of "Bibliography"
    cleaned_text = re.sub(r'((?:.*?\sBibliography\s){2}).*', r'\1', text, flags=re.DOTALL).replace("\n", "").replace("\r", "")

    #print(cleaned_text)
    return cleaned_text

def main():
    user_query = "Who is Peter Abelard?"#input("Input the query... ").strip()
    client = import_google_api()
    gemini_embedding_function = embedding_function(client)
    #pickled_dictionary, concatenated_list = unpickle()
    #create_collection(gemini_embedding_function, client, concatenated_list)
    embed_fn, collection = persistent_client(gemini_embedding_function)
    get_article(user_query, embed_fn, collection, client)
    summarize_article(user_query, embed_fn, collection, client)
    get_full_article(user_query, embed_fn, collection, client)
    
if __name__ == "__main__":
    main()
