from fastapi import FastAPI
from app.routes import sentiment_route

app = FastAPI()

# Include the text processing routes
app.include_router(sentiment_route.router)
