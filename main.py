from fastapi import FastAPI
from konlpy.tag import Komoran
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def read_root():
    return {"msg": "World"}


@app.get("/classes")
async def read_classes(data = None):
    result = Komoran().pos("안녕하세요 안녕")
    return {"classes": result}

hon_tokens = [word.rstrip('\n') for word in open('komoran_honorific_token.txt', 'r',encoding='utf-8')]
class counterRequest(BaseModel):
    text: str

@app.post("/predict", )
async def honorific_token_counter(request: counterRequest):
    cnt = 0
    for i in komoran.pos(request.text):
        if str(i) in hon_tokens:
            cnt += 1
    return {"cnt": cnt}