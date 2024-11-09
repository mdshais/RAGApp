from typing import Union

from fastapi import FastAPI

from main import getLLMAnswer

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welsome to my RAG APP API"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


@app.get("/llm")
def getAnswerFromLLM(q: Union[str, None] = None):
    response = getLLMAnswer(q)
    return {"response": response}