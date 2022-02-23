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