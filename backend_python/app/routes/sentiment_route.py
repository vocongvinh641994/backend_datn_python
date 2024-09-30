from exceptiongroup import catch
from fastapi import APIRouter, Request
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from fastapi.concurrency import run_in_threadpool
from .categorized_label_route import categorizedLabel

router = APIRouter()

# Path to your finetuned model directory
path_finetuned_phobert_attitude = "app/models/phobert_models/phobert_attitude/2024_09_30_03_08_08/"
path_finetuned_phobert_driver = "app/models/phobert_models/phobert_driver/2024_09_30_02_50_39"
path_finetuned_phobert_application = "app/models/phobert_models/phobert_application/checkpoint_20240812_epoch1"
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

model_attitude = AutoModelForSequenceClassification.from_pretrained(path_finetuned_phobert_attitude)
tokenizer_attitude = AutoTokenizer.from_pretrained(path_finetuned_phobert_attitude)
model_attitude.eval()  # Set the model to evaluation mode

model_driver = AutoModelForSequenceClassification.from_pretrained(path_finetuned_phobert_driver)
tokenizer_driver = AutoTokenizer.from_pretrained(path_finetuned_phobert_driver)
model_driver.eval()  # Set the model to evaluation mode

# Define label mapping
label_mapping = {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
}

label_mapping_title = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

sentiment_categories = {
    "Application": 'application',
    "Attitude": 'attitude',
    "Driver": 'driver'
}

# Determine if GPU is available and set the device accordingly
device = 0 if torch.cuda.is_available() else -1


async def run_pipeline(classifier, texts):
    # Run the pipeline in a thread pool to avoid blocking the event loop
    return await run_in_threadpool(classifier, texts)

async def predict_list(model, reviews):
    predicts = []
    try:
        inputs = tokenizer(
            reviews,
            return_tensors="pt",  # Return PyTorch tensors
            padding=True,  # Pad shorter reviews to the length of the longest
            truncation=True,  # Truncate longer reviews to max_length
            max_length=128  # Define the maximum sequence length
        )

        # Perform inference with the model for multiple reviews
        with torch.no_grad():
            outputs = model(**inputs)
        # Get raw logits and apply sigmoid activation
        logits = outputs.logits
        predictions = torch.sigmoid(logits)
        # Get the index of the largest value
        max_indices = torch.argmax(predictions, dim=1)
        predicts = max_indices.tolist()

    except Exception as e:
        predicts = []
    return predicts


@router.post("/classify_phobert/")
async def classify_phobert(request: Request):
    try:
        # Parse the incoming JSON request
        data = await request.json()
        reviews = data.get("reviews", [])

        # Validate 'reviews' field
        if not reviews or not isinstance(reviews, list):
            return {"error": "Invalid or missing 'reviews' field. Please provide a list of reviews."}

        # Filter reviews based on keywords
        reviews_categorized_label = await categorizedLabel(reviews)
        filter_unknown = [review for review in reviews_categorized_label if (
                    review["label_application"] == 0 and review["label_driver"] == 0 and review["label_attitude"] == 0)]

        filter_application = [review for review in reviews_categorized_label  if review["label_application"] == 1]
        filter_driver = [review for review in reviews_categorized_label  if review["label_driver"] == 1]
        filter_attitude = [review for review in reviews_categorized_label  if review["label_attitude"] == 1]
        # Create sentiment-analysis pipelines using GPU (if available)
        texts_application = [review["content"] for review in filter_application]
        texts_driver = [review["content"] for review in filter_driver]
        texts_attitude = [review["content"] for review in filter_attitude]

        classifier_application = pipeline("sentiment-analysis", model=path_finetuned_phobert_application,
                                          tokenizer=tokenizer, device=device)
        classifier_driver = await predict_list(model_driver, texts_driver)

        classifier_attitude = await predict_list(model_attitude, texts_attitude)

        # Perform sentiment analysis asynchronously
        result_application = await run_pipeline(classifier_application, texts_application) if texts_application else []
        # Map the labels to meaningful sentiment values
        # Iterate over result_application with index
        for index, r in enumerate(result_application):
            # r['sentiment'] = label_mapping.get(r['label'], "unknown")
            # r['sentimentName'] = label_mapping_title.get(r['label'], "unknown")
            r['reviewId'] = reviews_categorized_label[index]['id']
            r['reviewsCategory'] = sentiment_categories.get('Application')
            r['content'] = reviews_categorized_label[index]['content']
            print(f"Application - Index: {index}, Result: {r}")
        # return {"result": classifier_driver}
        # Iterate over result_driver with index
        for index, r in enumerate(filter_driver):
            sentiment = classifier_driver[index]
            r['sentiment'] = sentiment
            r['sentimentName'] = label_mapping_title.get(sentiment, "unknown")
            r['reviewId'] = reviews_categorized_label[index]['id']
            r['reviewsCategory'] = sentiment_categories.get('Driver')
            r['content'] = reviews_categorized_label[index]['content']
            print(f"Driver - Index: {index}, Result: {r}")

        # Iterate over result_attitude with index
        for index, r in enumerate(filter_attitude):
            sentiment = classifier_attitude[index]
            r['sentiment'] = sentiment
            r['sentimentName'] = label_mapping_title.get(sentiment, "unknown")
            r['reviewId'] = reviews_categorized_label[index]['id']
            r['reviewsCategory'] = sentiment_categories.get('Attitude')
            r['content'] = reviews_categorized_label[index]['content']
            print(f"Attitude - Index: {index}, Result: {r}")
        return {
            "result_application": result_application,
            "result_driver": filter_driver,
            "result_attitude": filter_attitude,
            "result_unknown": filter_unknown
        }

    except Exception as e:
        # Catch and return any exception
        return {"error": f"An error occurred: {str(e)}"}