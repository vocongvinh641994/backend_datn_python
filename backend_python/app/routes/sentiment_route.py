from fastapi import APIRouter, Request
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .categorized_label_route import categorized_label
from torch.cuda.amp import autocast
from app.models.openai import openai
from app.model.sentiment import sentiment_obj

router = APIRouter()

# Paths to your finetuned model directories
path_finetuned_phobert_evaluation = "app/models/phobert_models/phobert_evaluation/2025_02_03"

# Tokenizer (common for all models if using same tokenizer)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Load models
model_evaluation = AutoModelForSequenceClassification.from_pretrained(path_finetuned_phobert_evaluation)

# Set models to evaluation mode
model_evaluation.eval()

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

        # Get the column index of the max value in each row
        max_positions = logits.argmax(dim=1).tolist()

        # Collect predictions instead of printing
        all_preds.extend(max_positions)
    return all_preds

def get_category(application, driver, operator):
    if application>0 and driver > 0 and operator >0:
        return sentiment_obj.APPLICATION_DRIVER_OPERATOR
    if driver>0 and operator >0:
        return sentiment_obj.DRIVER_OPERATOR
    if application>0 and operator>0:
        return sentiment_obj.APPLICATION_OPERATOR
    if application>0 and driver>0:
        return sentiment_obj.APPLICATION_DRIVER
    if operator>0:
        return sentiment_obj.OPERATOR
    if driver>0:
        return sentiment_obj.DRIVER
    if application>0:
        return sentiment_obj.APPLICATION
    return sentiment_obj.UNKNOWN

@router.post("/classify/")
async def classify(request: Request):
    # try:
        # Parse the incoming JSON request
        data = await request.json()
        reviews = data.get("reviews", [])
        isOpenAI = data.get("isOpenAI", False)
        if(isOpenAI):
            responseOpenAI = await openai.categorize(reviews)
            for index, review in enumerate(reviews):
                sentiment_obj = responseOpenAI[index]
                application = sentiment_obj['application']
                driver = sentiment_obj['driver']
                operator = sentiment_obj['operator']
                category = get_category(application, driver, operator)
                review['application_sentiment'] = sentiment_obj['application_sentiment']
                review['driver_sentiment'] = sentiment_obj['driver_sentiment']
                review['operator_sentiment'] = sentiment_obj['operator_sentiment']
                review['category'] = category

            # [{'id': 1206, 'application': 1, 'driver': 1, 'operator': 1, 'application_sentiment': 2,
            #  'driver_sentiment': 2, 'operator_sentiment': 2}]

            return   reviews if reviews else []
        else:
            # Validate 'reviews' field
            if not reviews or not isinstance(reviews, list):
                return {"error": "Invalid or missing 'reviews' field. Please provide a list of reviews."}
                # Categorize reviews based on labels (using an external categorizedLabel function)
            reviews_categorized_label = await categorized_label(reviews)

            # Extract review contents for each category
            texts = [review["content"] for review in reviews]
            # Predict sentiment in parallel for each category
            classifier_application = await  predict_list_batched(model_evaluation, texts)
            # ✅ Print once after getting predictions
            # Assign predictions back to the original reviews for Application
            for index, review in enumerate(reviews_categorized_label):
                sentiment = classifier_application[index]
                if review['application'] == 1:
                    review['application_sentiment'] = sentiment
                if review['driver'] == 1:
                    review['driver_sentiment'] = sentiment
                if review['operator'] == 1:
                    review['operator_sentiment'] = sentiment
                if 'id' in reviews[index]:
                    review['reviewId'] = reviews[index]['id']
                category = get_category(review['application'], review['driver'], review['operator'])
                review['category'] = category

            print("111111111:")
            print(reviews_categorized_label)
            print("222222222:")

            return  reviews_categorized_label if reviews_categorized_label else []
    # except Exception as e:
    #     # Handle and return any exception
    #     return {"error": f"An error occurred: {str(e)}"}
