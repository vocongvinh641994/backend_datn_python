from fastapi import FastAPI
from app.routes import sentiment_route

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
# Include the text processing routes
app.include_router(sentiment_route.router)
