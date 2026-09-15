import chromadb
import os

from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY not found.")
    exit()

# Create clients
chroma_client = chromadb.Client()
gemini_client = genai.Client(api_key=api_key)


# Create collection
collection = chroma_client.get_or_create_collection(
    name="learning_documents"
)


# Documents
documents = [
    "Supervised learning uses labeled data to train a model.",
    "Unsupervised learning uses unlabeled data to discover patterns.",
    "Neural networks are computing systems inspired by biological brains.",
    "Chocolate cake is made using ingredients such as flour, sugar, and eggs.",
    "Thomas Edison invented the practical electric light bulb.",
    "Alexander Graham Bell is commonly credited with inventing the telephone."
]


# IDs
ids = [
    "chunk_1",
    "chunk_2",
    "chunk_3",
    "chunk_4",
    "chunk_5",
    "chunk_6"
]


# Metadata
metadatas = [
    {
        "source": "machine_learning.txt",
        "topic": "supervised_learning",
        "page": 1
    },
    {
        "source": "machine_learning.txt",
        "topic": "unsupervised_learning",
        "page": 2
    },
    {
        "source": "neural_networks.txt",
        "topic": "neural_networks",
        "page": 1
    },
    {
        "source": "cooking.txt",
        "topic": "cooking",
        "page": 3
    },
    {
        "source": "inventions.txt",
        "topic": "electricity",
        "page": 4
    },
    {
        "source": "inventions.txt",
        "topic": "telephone",
        "page": 5
    }
]


# Store documents in Chroma
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print("Documents stored:", collection.count())


# User question
query = "Who invented the telephone?"

# Retrieve relevant document
results = collection.query(
    query_texts=[query],
    n_results=1
)


# Extract retrieved information
context = results["documents"][0][0]
source = results["metadatas"][0][0]["source"]
page = results["metadatas"][0][0]["page"]


print("\nRetrieved context:")
print(context)

print("\nSource:")
print(source)

print("Page:")
print(page)


# Create RAG prompt
prompt = f"""
You are a document question-answering assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{query}

Answer:
"""


# Send context + question to Gemini
response = gemini_client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
)


# Get answer
answer = response.output_text


print("\nAnswer:")
print(answer)