from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import data

url = "https://www.apple.com/shop/buy-iphone/iphone-duo"

web_loader = WebBaseLoader(url)
data = web_loader.load()
print(data[0].page_content)