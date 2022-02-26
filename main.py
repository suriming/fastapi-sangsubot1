from fastapi import FastAPI
from konlpy.tag import Komoran, Mecab
from pydantic import BaseModel


from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle

from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://localhost",
    "http://3.38.106.209"
    "http://3.38.106.209:80"
    "http://3.38.106.209:8080"
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
class counterResponse(BaseModel):
    cnt: str

komoran = Komoran(userdic='/content/sangsu_dict.txt')

@app.post("/predict", )
async def honorific_token_counter(request: counterRequest):
    cnt = 0
    for i in komoran.pos(request.text):
        if str(i) in hon_tokens:
            cnt += 1
    return counterResponse(
        cnt = cnt
    )

class sentimentRequest(BaseModel):
    text: str

@app.post("/sentiment", )
async def sentiment_predict(request: sentimentRequest):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    target_sentence = request.text

    max_len = 80
    model = load_model('best_model.h5')
    stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯',
                 '지', '임', '게', '이', '있', '하', '것', '들', '그', '되', '수', '이', '보', '나', '사람', '주', '아니', '등', '같', '우리',
                 '때', '가', '한', '지', '대하', '오', '말', '일', '그렇', '위하', '때문', '그것', '두', '알', '그러나', '받', '일', '그런', '또',
                 '사회', '많', '그리고', '중', '따르', '만들', '지금', '고', '다']  # 불용어

    target_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', target_sentence)
    target_sentence = Mecab().morphs(target_sentence)
    # target_sentence = [word for word in target_sentence if not word in stopwords]
    encoded = tokenizer.texts_to_sequences([target_sentence])
    pad_new = pad_sequences(encoded, maxlen=max_len)
    score = model.predict(pad_new)
    test = score[0].tolist()
    test = dict(zip(["1", "2", "3", "4", "5"], test))
    # output_score = []
    # for idx in range(5) :
    #     output_score.append(score[0][idx])
    # outputdict = dict(enumerate(output_score))
    return {"score": test}  