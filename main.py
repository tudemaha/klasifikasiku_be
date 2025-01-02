from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from classification import Classification
from response import Response
from dotenv import load_dotenv
from os import getenv

load_dotenv()
app = FastAPI()

origins = getenv("ALLOWED_ORIGINS").split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/public", StaticFiles(directory="public"), name="public")
app.mount("/classification", StaticFiles(directory="classification"), name="classification")

@app.get("/")
def read_root():
    return {"message": "Hello World!"}

@app.post('/classification')
async def classify(transaction: UploadFile):
    res = Response.bad_request(error=["The uploaded file is not an Excel file."])
    if not transaction.filename.endswith((".xls", ".xlsx")):
        return JSONResponse(res, res.get("status"))
    
    content = await transaction.read()
    try:
        classification = Classification(content)
        transactions, pred, f1, recall, precision = classification.knn_classification()
        transactions['prediction'] = pred

        id = str(uuid.uuid4())
        transactions.to_excel('classification/{}.xlsx'.format(id), index=False, sheet_name='knn_classification')

        result = transactions.iloc[:, [0, 1, 5]].to_dict(orient='records')

        res = Response.success(data={
            'id': id,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'excel_download': '{}/classification/{}.xlsx'.format(getenv("HOST"), id),
            'result': result
        })
        return JSONResponse(res, res.get("status"))
    except Exception as e:
        res = Response.internal_error(error=[e])
        return JSONResponse(res, res.get("status"))