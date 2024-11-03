from fastapi import APIRouter, Request
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi.concurrency import run_in_threadpool
from .categorized_label_route import categorizedLabel
from asyncio import gather
from torch.cuda.amp import autocast

router = APIRouter()

# Paths to your finetuned model directories
path_finetuned_phobert_attitude = "app/models/phobert_models/phobert_attitude/2024_10_31"
path_finetuned_phobert_driver = "app/models/phobert_models/phobert_driver/2024_10_31"
path_finetuned_phobert_application = "app/models/phobert_models/phobert_application/2024_10_31"
path_finetuned_phobert_operator = "app/models/phobert_models/phobert_operator/2024_10_31"
path_finetuned_phobert_interior = "app/models/phobert_models/phobert_interior/2024_10_31"

# Tokenizer (common for all models if using same tokenizer)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Load models
model_attitude = AutoModelForSequenceClassification.from_pretrained(path_finetuned_phobert_attitude)
model_driver = AutoModelForSequenceClassification.from_pretrained(path_finetuned_phobert_driver)
model_application = AutoModelForSequenceClassification.from_pretrained(path_finetuned_phobert_application)
model_operator = AutoModelForSequenceClassification.from_pretrained(path_finetuned_phobert_operator)
model_interior = AutoModelForSequenceClassification.from_pretrained(path_finetuned_phobert_interior)

# Set models to evaluation mode
model_attitude.eval()
model_driver.eval()
model_application.eval()
model_operator.eval()
model_interior.eval()

# Define label mappings
label_mapping_title = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
sentiment_categories = {"Application": 'application', "Attitude": 'attitude', "Driver": 'driver', "Operator": 'operator', "Interior": 'interior'}

# Determine if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move models to the appropriate device (GPU or CPU)
model_attitude.to(device)
model_driver.to(device)
model_application.to(device)
model_operator.to(device)
model_interior.to(device)


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


async def predict_all(model_application, model_driver, model_attitude, model_operator, model_interior, texts_application, texts_driver, texts_attitude, texts_operator, texts_interior):
    result_application, result_driver, result_attitude, result_operator, result_interior = await gather(
        predict_list_batched(model_application, texts_application),
        predict_list_batched(model_driver, texts_driver),
        predict_list_batched(model_attitude, texts_attitude),
        predict_list_batched(model_operator, texts_operator),
        predict_list_batched(model_interior, texts_interior)
    )
    return result_application, result_driver, result_attitude, result_operator, result_interior


@router.post("/classify_phobert/")
async def classify_phobert(request: Request):
    try:
        # Parse the incoming JSON request
        data = await request.json()
        reviews = data.get("reviews", [])

        # Validate 'reviews' field
        if not reviews or not isinstance(reviews, list):
            return {"error": "Invalid or missing 'reviews' field. Please provide a list of reviews."}

        # Categorize reviews based on labels (using an external categorizedLabel function)
        reviews_categorized_label = await categorizedLabel(reviews)

        filter_unknown = [review for review in reviews_categorized_label if (
                review["label_application"] == 0 and review["label_driver"] == 0 and review["label_attitude"] == 0 and review["label_operator"] == 0 and review["label_interior"] == 0)]

        filter_application = [review for review in reviews_categorized_label if review["label_application"] >= 1]
        filter_driver = [review for review in reviews_categorized_label if review["label_driver"] >= 1]
        filter_attitude = [review for review in reviews_categorized_label if review["label_attitude"] >= 1]
        filter_operator = [review for review in reviews_categorized_label if review["label_operator"] >= 1]
        filter_interior = [review for review in reviews_categorized_label if review["label_interior"] >= 1]

        # Extract review contents for each category
        texts_application = [review["content"] for review in filter_application]
        texts_driver = [review["content"] for review in filter_driver]
        texts_attitude = [review["content"] for review in filter_attitude]
        texts_operator = [review["content"] for review in filter_operator]
        texts_interior = [review["content"] for review in filter_interior]

        # Predict sentiment in parallel for each category
        classifier_application, classifier_driver, classifier_attitude, classifier_operator, classifier_interior = await predict_all(
            model_application, model_driver, model_attitude, model_operator, model_interior, texts_application, texts_driver, texts_attitude, texts_operator, texts_interior
        )

        # Assign predictions back to the original reviews for Application
        for index, review in enumerate(filter_application):
            sentiment = classifier_application[index]
            review['sentiment'] = sentiment
            review['sentimentName'] = label_mapping_title.get(sentiment, "unknown")
            review['reviewId'] = reviews_categorized_label[index]['id']
            review['reviewsCategory'] = sentiment_categories.get('Application')
            review['content'] = reviews_categorized_label[index]['content']

        # Assign predictions back to the original reviews for Driver
        for index, review in enumerate(filter_driver):
            sentiment = classifier_driver[index]
            review['sentiment'] = sentiment
            review['sentimentName'] = label_mapping_title.get(sentiment, "unknown")
            review['reviewId'] = reviews_categorized_label[index]['id']
            review['reviewsCategory'] = sentiment_categories.get('Driver')
            review['content'] = reviews_categorized_label[index]['content']

        # Assign predictions back to the original reviews for Attitude
        for index, review in enumerate(filter_attitude):
            sentiment = classifier_attitude[index]
            review['sentiment'] = sentiment
            review['sentimentName'] = label_mapping_title.get(sentiment, "unknown")
            review['reviewId'] = reviews_categorized_label[index]['id']
            review['reviewsCategory'] = sentiment_categories.get('Attitude')
            review['content'] = reviews_categorized_label[index]['content']

        for index, review in enumerate(filter_operator):
            sentiment = classifier_operator[index]
            review['sentiment'] = sentiment
            review['sentimentName'] = label_mapping_title.get(sentiment, "unknown")
            review['reviewId'] = reviews_categorized_label[index]['id']
            review['reviewsCategory'] = sentiment_categories.get('Attitude')
            review['content'] = reviews_categorized_label[index]['content']

        for index, review in enumerate(filter_interior):
            sentiment = classifier_interior[index]
            review['sentiment'] = sentiment
            review['sentimentName'] = label_mapping_title.get(sentiment, "unknown")
            review['reviewId'] = reviews_categorized_label[index]['id']
            review['reviewsCategory'] = sentiment_categories.get('Attitude')
            review['content'] = reviews_categorized_label[index]['content']

        # Return the categorized results
        return {
            "result_application": filter_application,
            "result_driver": filter_driver,
            "result_attitude": filter_attitude,
            "result_operator": filter_operator,
            "result_interior": filter_interior,
            "result_unknown": filter_unknown
        }

    except Exception as e:
        # Handle and return any exception
        return {"error": f"An error occurred: {str(e)}"}