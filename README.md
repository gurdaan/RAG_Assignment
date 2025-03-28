# News RAG Application with LinkedIn Post Generator

A Retrieval-Augmented Generation (RAG) system that answers queries by retrieving relevant news articles and generating contextual summaries, with an integrated LinkedIn Post Generator.

## Features

- **News Retrieval**: Fetch relevant news articles based on user queries
- **Contextual Summarization**: Generate accurate summaries of retrieved news
- **LinkedIn Post Generator**: Create shareable LinkedIn posts from news content
- **Weaviate Vector Database**: Efficient storage and retrieval of news embeddings
- **Autogen Agents**: AI-powered agents for processing and generation tasks

## Steps To Run The Project
1) Run pip install -r requirements.txt
2) Run docker compose up to start weaviate and olama
3) Run data.py to get the data and save in Weaviate DB
4) Autogen Agents are present in Agents Folder
5) To run the project run uvicorn main:app
