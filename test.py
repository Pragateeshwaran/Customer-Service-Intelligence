from main import csi
from langchain_groq import ChatGroq
import dotenv
import os

dotenv.load_dotenv()
llm = ChatGroq(model_name="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
app = csi("llama3-70b-8192", llm)
print(app.process(path='new.wav'))
