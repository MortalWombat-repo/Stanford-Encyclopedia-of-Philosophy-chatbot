import subprocess
from google import genai
from google.genai import types
from IPython.display import Markdown
from dotenv import load_dotenv
import os

# Uninstall conflicting packages quietly and without confirmation
subprocess.run(["pip", "uninstall", "-qqy", "jupyterlab", "kfp"], check=True)

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

