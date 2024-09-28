from app.routes import sentiment_route
from app.routes import categorized_label_route

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
# Include the text processing routes
app.include_router(sentiment_route.router)
app.include_router(categorized_label_route.router)
