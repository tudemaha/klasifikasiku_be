from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/public", StaticFiles(directory="public"), name="public")
app.mount("/classification", StaticFiles(directory="classification"), name="classification")

@app.get("/")
def read_root():
    return {"Hello": "World"}
