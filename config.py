import os
from dotenv import load_dotenv

load_dotenv()
BARD_KEY = os.getenv('GOOGLE_API_KEY')