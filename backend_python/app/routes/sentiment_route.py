from fastapi import APIRouter, Request
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .categorized_label_route import categorized_label
from asyncio import gather
from torch.cuda.amp import autocast

router = APIRouter()

# Paths to your finetuned model directories
path_finetuned_phobert_evaluation = "app/models/phobert_models/phobert_evaluation/2025_02_03"

# Tokenizer (common for all models if using same tokenizer)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Load models
model_evaluation = AutoModelForSequenceClassification.from_pretrained(path_finetuned_phobert_evaluation)

# Set models to evaluation mode
model_evaluation.eval()

# Define label mappings
label_mapping_title = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
sentiment_categories = {"Application": 'application', "Attitude": 'attitude', "Driver": 'driver', "Operator": 'operator', "Interior": 'interior'}

# Determine if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move models to the appropriate device (GPU or CPU)
model_evaluation.to(device)


async def predict_list_batched(model, reviews, batch_size=32):
    all_preds = []
    for i in range(0, len(reviews), batch_size):
        batch_reviews = reviews[i:i + batch_size]
        inputs = tokenizer(
            batch_reviews,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        # Use mixed precision if possible
        with autocast(), torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predictions = torch.softmax(logits, dim=1)  # Apply softmax across the classes for each sample
        max_indices = torch.argmax(predictions, dim=1).tolist()
        all_preds.extend(max_indices)
    return all_preds

@router.post("/classify_phobert/")
async def classify_phobert(request: Request):
    # try:
        # Parse the incoming JSON request
        data = await request.json()
        reviews = data.get("reviews", [])
        # Validate 'reviews' field
        if not reviews or not isinstance(reviews, list):
            return {"error": "Invalid or missing 'reviews' field. Please provide a list of reviews."}
        # Categorize reviews based on labels (using an external categorizedLabel function)
        reviews_categorized_label = await categorized_label(reviews)
        # Extract review contents for each category
        texts = [review["content"] for review in reviews_categorized_label]
        # Predict sentiment in parallel for each category
        classifier_application = await  predict_list_batched(model_evaluation, texts)
        # Assign predictions back to the original reviews for Application
        for index, review in enumerate(reviews):
            sentiment = classifier_application[index]
            review['sentiment'] = sentiment
            review['sentimentName'] = label_mapping_title.get(sentiment, "unknown")
            if 'id' in reviews_categorized_label[index]:
                review['reviewId'] = reviews_categorized_label[index]['id']
            review['content'] = reviews_categorized_label[index]['content']
            category = "unknown"
            if review["label_application"] == 1:
                category = "application"
            if review["label_driver"] == 1:
                category = "driver"
            if review["label_attitude"] == 1:
                category = "attitude"
            if review["label_interior"] == 1:
                category = "interior"
            if review["label_operator"] == 1:
                category = "operator"
            review["category"] = category
        return  reviews if reviews else []
    # except Exception as e:
    #     # Handle and return any exception
    #     return {"error": f"An error occurred: {str(e)}"}
