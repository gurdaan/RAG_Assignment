from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from Agents import agent
from datetime import datetime

app = FastAPI()

# CORS (For frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query_agent_end(request: Request):
    data = await request.json()
    query = data.get('query')
    if not query:
        return {"error": "No query provided"}
    news_info, linkedin_info = await agent.query_agent(query)
    result = "News:\n " + news_info + "\nLinkedIn:\n" + linkedin_info
    return {"data": result}

