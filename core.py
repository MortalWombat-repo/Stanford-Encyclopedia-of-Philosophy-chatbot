import subprocess
from google import genai
from google.genai import types
from IPython.display import Markdown

# Uninstall conflicting packages quietly and without confirmation
subprocess.run(["pip", "uninstall", "-qqy", "jupyterlab", "kfp"], check=True)

