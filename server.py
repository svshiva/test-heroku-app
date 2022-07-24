from fastapi import FastAPI, Request, Form, UploadFile, File
from helper import *



app=FastAPI()

@app.get('/')
def root():
    return {'message': 'Hello World'}

@app.post('/api/face-validation')
async def face_validation(test_image: UploadFile, id: str = Form(default="Null"), image: str = Form(default="Null")):
    test_image = await test_image.read()
    test_image = byteImg_to_cvImg(test_image)
    result,distance = predict(get_image_from_url(image), test_image)
    result = bool(result[0])
    distance = str(distance[0])
    return {'id': id, 'result': result, 'distance': distance}

