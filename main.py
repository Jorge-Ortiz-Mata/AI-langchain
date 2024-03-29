import base64
import requests
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.prompts import PromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# === Create responses based on Power Point Presentations
# pip install python-pptx
def presentation_pp():
  loader = UnstructuredPowerPointLoader("./sistema-solar.pptx")
  embeddings = OpenAIEmbeddings()
  question = "¿Que es la tierra?"

  text_splitter_config = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=400,
    chunk_overlap=100
  )

  docs = loader.load_and_split(
    text_splitter=text_splitter_config
  )

  db = Chroma.from_documents(
    docs, 
    embedding=embeddings
  )

  results = db.similarity_search(question)

  top_result = results[0].page_content

  llm = ChatOpenAI()

  prompt = PromptTemplate(
    template="""
      Por favor, crea una respuesta hacia el usuario en base a la informacion que recibes. Recibiras la pregunta del usuario y un texto que proviene de una presentacion
      de power point donde muestra la respuesta que mas se asimila a la pregunta del usuario.
      Pregunta: {question}
      Texto extraido: {top_result}
      Crea la respuesta en base al texto extraido que recibes. Si el texto no contiene informacion acerca de la pregunta del usuario, por favor, indicale al usuario
      que en base al texto extraido, no pudiste encontrar la respuesta. No busques la respuesta en ningun otro lugar.
      Trata de crear las respuestas lo mas cortas posibles
    """,
    input_variables=["top_result", "question"]
  )

  chain = LLMChain(llm=llm, prompt=prompt)

  return chain({"top_result": top_result, "question": question})

# === Create images based on small descriptions
def image_generator(): 
  llm = ChatOpenAI(temperature=0.9)

  prompt = PromptTemplate(
      input_variables=["image_desc"],
      template="Generate a detailed prompt to generate an image based on the following description: {image_desc}. The prompt must be short",
  )

  chain = LLMChain(llm=llm, prompt=prompt)

  image_url = DallEAPIWrapper().run(chain.run({"image_desc": "a man drinking a coffe"}))

  return image_url

# === Create responses based on images
def image_url_interpreter():
  llm = OpenAI()

  response = llm.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What’s in this image?"},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
              "detail": "auto",
            },
          },
        ],
      }
    ],
    max_tokens=300,
  )

  return response 

# === Image from current path
def image_path_interpretation():
  def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')
    
  image_path = "videogames.jpg"

  base64_image = encode_image(image_path)

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
  }

  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What’s in this image? Create a short response where possible"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  return response.json()

def image_generation_dalle():
  client = OpenAI()

  response = client.images.generate(
    model="dall-e-3",
    prompt="an image of new york city that captures the feeling of nostalgia for the 1980s, featuring iconic objects from that era.",
    size="1024x1024",
    quality="standard",
    n=1,
  )

  return response.data[0].url

res = image_generation_dalle()
print(res)