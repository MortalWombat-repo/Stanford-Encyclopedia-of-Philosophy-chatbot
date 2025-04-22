import subprocess
from google import genai
from google.genai import types
from IPython.display import Markdown
from dotenv import load_dotenv
import os

# Uninstall conflicting packages quietly and without confirmation
subprocess.run(["pip", "uninstall", "-qqy", "jupyterlab", "kfp"], check=True)

from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry

from google.genai import types

"""
# Load variables from .env file
load_dotenv()

# Access the value
google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    print("API key inputted")

client = genai.Client(api_key=GOOGLE_API_KEY)

for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)
"""
GOOGLE_API_KEY = ""

client = genai.Client(api_key=GOOGLE_API_KEY)

for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)

if not os.path.exists("output/chroma"):
    print("Folder not found. Running setup.")
    import zipfile
    import os

    zip_path = "output/chroma_db.zip"
    extract_path = "output/chroma"  #ChromaDB folder

    # Unzip it
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

import chromadb

# Load the persistent ChromaDB from the extracted path
db = chromadb.PersistentClient(path="output/chroma")

# List collection names to verify the database
collection_names = db.list_collections()
#print("Collection names:", collection_names)

# Access a specific collection by its name
collection_name = collection_names[0]  # You can select the first collection or any other collection by name
collection = db.get_collection(collection_name)

# Peek into the collection (view first item)
#print(collection.peek(1))
#print(collection.count())
print(collection.metadata)  # Check metadata for clues about the embedding model
print(collection.count())  # Verify the collection has data

# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/embedding-004",#"models/text-embedding-005",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]

embed_fn = GeminiEmbeddingFunction()

embed_fn = GeminiEmbeddingFunction()
embeddings = embed_fn(["test input"])
print("Embedding dimensionality:", len(embeddings[0]))

# Switch to query mode when generating embeddings.
embed_fn.document_mode = False

# Search the Chroma DB using the specified query.
query = "Explain Abelards Logic."  #< --- Type your question here

result = collection.query(query_texts=[query], n_results=1)
print(result)


'''
[all_passages] = result["documents"]

Markdown(all_passages[0])

query_oneline = query.replace("\n", " ")

# This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
prompt = """You are a helpful and informative bot that answers questions using text from the reference passage included below. You answer with the knowledge of a philosophy postdoc that is trying to bridge complicated subjects to a non-philosophy knowing audience, so be sure to break down complicated concepts and strike a friendly and converstional tone.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
If the passage is irrelevant to the answer, you may ignore it.

QUESTION: {query_oneline}
"""

# Add the retrieved documents to the prompt.
for passage in all_passages:
    passage_oneline = passage.replace("\n", " ")
    prompt += f"PASSAGE: {passage_oneline}\n"

#print(prompt)

answer = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt)

Markdown(answer.text)
'''


