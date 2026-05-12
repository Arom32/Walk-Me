from fastapi import FastAPI
from app.config import settings

app = FastAPI(title="Worme API")

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "database_host": settings.DB_HOST,
        "database_name": settings.DB_NAME
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)