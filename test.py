from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

response = llm.invoke("대한관광스키지도자연맹 챗봇 테스트")

print(response.content)