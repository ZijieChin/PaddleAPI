import os

import uvicorn
from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR

app = FastAPI()
paddle = PaddleOCR(use_gpu=False, use_angle_cls=False)


@app.post('/paddle/v1/img')
async def rec(upfile: UploadFile = File(...)):
    res = await upfile.read()
    filepath = 'tmp/' + upfile.filename
    try:
        with open(filepath, 'wb') as f:
            f.write(res)
        result = paddle.ocr(filepath)
        output = ''
        for text in result:
            output += text[1][0]
        success = True
    except Exception as e:
        output = str(e)
        success = False
    os.remove(filepath)
    return {
        'success': success,
        'data': output
    }


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=8086, workers=1)
