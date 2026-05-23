# KSTIA AI Assistant

LangChain 기반 대한관광스키지도자연맹(KSTIA) AI 챗봇 프로젝트

## Overview

KSTIA AI Assistant는 대한관광스키지도자연맹의 자격증, 강습 과정, 행사 일정 및 FAQ 정보를 자연어 기반으로 검색하고 응답하는 RAG(Retrieval-Augmented Generation) 기반 챗봇 프로젝트입니다.

기존 연맹 운영 과정에서 반복적으로 발생하던 문의(자격증 일정, 강습 과정, 준비물, 행사 정보 등)를 효율적으로 안내하기 위해 개발하였습니다.

본 프로젝트는 단순한 LLM 응답이 아닌, Markdown 기반 연맹 문서를 벡터화하여 검색 후 응답하는 구조를 목표로 합니다.

---

## Features

- KSTIA FAQ 기반 질의응답
- Markdown 문서 기반 정보 검색
- FAISS 벡터 데이터베이스 활용
- LangChain 기반 RAG 파이프라인
- OpenAI GPT API 연동
- Streamlit 기반 챗봇 UI
- 자연어 기반 사용자 질의 처리

---

## Example Questions

- "레벨2 검정 일정 알려줘"
- "스키 처음 타는데 지원 가능한가요?"
- "강습 준비물은 무엇인가요?"
- "관광스키지도자 자격증은 어떻게 취득하나요?"
- "다음 시즌 행사 일정 알려줘"

---

## Tech Stack

### Backend
- Python 3.11
- LangChain
- LangGraph
- OpenAI API

### Vector Database
- FAISS

### Frontend
- Streamlit

### Data
- Markdown Knowledge Base

---

## Architecture

```text
User Question
    ↓
Document Retrieval (FAISS)
    ↓
Relevant Context Extraction
    ↓
OpenAI GPT Response Generation
    ↓
Final Answer
```

---

## Project Structure

```text
kstia-chatbot/
├── app.py
├── ingest.py
├── data/
│   └── kstia.md
├── vectorstore/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Knowledge Base Construction

연맹 관련 정보를 Markdown 형식으로 정리합니다.

예시:
- 자격증 안내
- 검정 절차
- 강습 과정
- 행사 일정
- FAQ

---

### 2. Document Embedding

문서를 Chunk 단위로 분할한 뒤 OpenAI Embedding 모델을 통해 벡터화합니다.

---

### 3. Vector Search

사용자의 질문과 가장 유사한 문서를 FAISS에서 검색합니다.

---

### 4. Response Generation

검색된 문서를 Context로 활용하여 GPT가 답변을 생성합니다.

---

## Motivation

생성형 AI는 단순한 대화 도구를 넘어, 실제 운영 환경의 정보 접근성과 사용자 경험을 개선할 수 있다고 생각했습니다.

본 프로젝트는 실제 스키 연맹 운영 과정에서 발생하는 반복 문의를 해결하기 위한 시도이며, AI 기반 문서 검색 시스템과 RAG 구조를 직접 구현해보기 위해 시작되었습니다.

---

## Future Improvements

- 실시간 공지사항 연동
- 일정 자동 업데이트 기능
- 다중 문서 검색
- 사용자별 맞춤 응답
- 음성 기반 인터페이스
- LangGraph 기반 Agent Workflow 확장

---

## Demo

Coming Soon

---

## Author

Kwangwoon University Computer Engineering

AI / Computer Vision / LLM Application Development
