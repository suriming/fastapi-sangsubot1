from fastapi import FastAPI
from konlpy.tag import Komoran

app = FastAPI()

@app.get("/")
async def read_root():
    return {"msg": "World"}


@app.get("/classes")
async def read_classes(data = None):
    result = Komoran().pos("안녕하세요 안녕")
    return {"classes": result}

hon_tokens = [word.rstrip('\n') for word in open('komoran_honorific_token.txt', 'r',encoding='utf-8')]

@app.post("/predict", response_model=counterResponse)
async def honorific_token_counter(request: counterRequest):
    cnt = 0
    for i in komoran.pos(request.text):
        if str(i) in hon_tokens:
            cnt += 1
    return counterResponse(
      cnt = cnt
    )