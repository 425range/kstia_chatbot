import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(
    page_title="KSTIA AI Assistant",
    page_icon="🎿",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f4f8ff 0%, #ffffff 100%);
}

.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

.kstia-header {
    text-align: center;
    padding: 24px 16px;
    border-radius: 24px;
    background: white;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    margin-bottom: 24px;
}

.kstia-title {
    font-size: 34px;
    font-weight: 800;
    color: #0b5ed7;
    margin-top: 12px;
}

.kstia-subtitle {
    font-size: 16px;
    color: #555;
    margin-top: 8px;
}

.stButton > button {
    border-radius: 999px;
    border: 1px solid #0b5ed7;
    color: #0b5ed7;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #0b5ed7;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="kstia-header">
    <div style="font-size:52px;">🎿</div>
    <div class="kstia-title">KSTIA AI Assistant</div>
    <div class="kstia-subtitle">
        대한관광스키지도자연맹 문서 기반 AI 안내 챗봇
    </div>
</div>
""", unsafe_allow_html=True)

# 0. Chat record reset
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 1. Load the vector store from disk
embeddings = OpenAIEmbeddings()

if not os.path.exists("vectorstore/index.faiss"):
    loader = TextLoader("data/kstia.md", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore")
else:
    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 2. Create the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# 3. Create the prompt template
prompt = ChatPromptTemplate.from_template("""
당신은 대한관광스키지도자연맹(KSTIA) 안내 챗봇입니다.

아래 참고 문서를 기반으로만 답변하세요.
문서에 없는 내용은 추측하지 말고 "제공된 문서에서는 확인할 수 없습니다."라고 답변하세요.

[참고 문서]
{context}

[사용자 질문]
{question}

[답변]
""")

# 4. Merge the searched documents into the prompt
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 5. Create RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# 6. Example of question
st.markdown("### 예시 질문")

example_questions = [
    "KSTIA는 어떤 단체인가요?",
    "스키를 처음 타도 참여할 수 있나요?",
    "KSTIA의 주요 활동은 무엇인가요?",
    "공식 인스타그램은 어디인가요?"
]

cols = st.columns(2)

for i, example in enumerate(example_questions):
    with cols[i % 2]:
        if st.button(example):
            st.session_state.selected_question = example

question = st.chat_input("궁금한 내용을 입력하세요")

if "selected_question" in st.session_state:
    question = st.session_state.selected_question
    del st.session_state.selected_question

# 7. Handle user input and generate response
if question:
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user", "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant", avatar="assets/관냥이.png"):
        with st.spinner("답변 생성 중..."):
            docs = retriever.invoke(question)
            context = format_docs(docs)

            messages = prompt.invoke({
                "context": context,
                "question": question
            })

            response = llm.invoke(messages)
            answer = response.content

            st.write(response.content)

            with st.expander("참고 문서 보기"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**참고 문서 {i+1}:**")
                    st.write(doc.page_content)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer
            }) 