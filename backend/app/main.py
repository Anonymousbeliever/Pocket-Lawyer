from fastapi import FastAPI

app = FastAPI(
    title="Pocket Lawyer API",
    version="0.1.0",
)

@app.get("/")
def root():
    return{
        "name": "Pocket Lawyer",
        "status": "online",
        "version": "0.1.0"
    }