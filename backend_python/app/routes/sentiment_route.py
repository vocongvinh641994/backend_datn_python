from fastapi import APIRouter, Request
from transformers import pipeline, AutoTokenizer
import torch
from fastapi.concurrency import run_in_threadpool
from .categorized_label_route import categorizedLabel

router = APIRouter()

# Path to your finetuned model directory
path_finetuned_phobert_attitude = "app/models/phobert_models/phobert_attitude/checkpoint_20240812_epoch1"
path_finetuned_phobert_driver = "app/models/phobert_models/phobert_driver/checkpoint_20240819_epoch5"
path_finetuned_phobert_application = "app/models/phobert_models/phobert_application/checkpoint_20240812_epoch1"
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Define label mapping
label_mapping = {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
}

label_mapping_title = {
    "LABEL_0": 'Negative',
    "LABEL_1": 'Neutral',
    "LABEL_2": 'Positive'
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
        classifier_driver = pipeline("sentiment-analysis", model=path_finetuned_phobert_driver, tokenizer=tokenizer,
                                     device=device)
        classifier_attitude = pipeline("sentiment-analysis", model=path_finetuned_phobert_attitude, tokenizer=tokenizer,
                                       device=device)


        # Perform sentiment analysis asynchronously
        result_application = await run_pipeline(classifier_application, texts_application) if texts_application else []
        result_driver = await run_pipeline(classifier_driver, texts_driver) if texts_driver else []
        result_attitude = await run_pipeline(classifier_attitude, texts_attitude) if texts_attitude else []

        # Map the labels to meaningful sentiment values
        # Iterate over result_application with index
        for index, r in enumerate(result_application):
            r['sentiment'] = label_mapping.get(r['label'], "unknown")
            r['sentimentName'] = label_mapping_title.get(r['label'], "unknown")
            r['reviewId'] = reviews_categorized_label[index]['id']
            r['reviewsCategory'] = sentiment_categories.get('Application')
            r['content'] = reviews_categorized_label[index]['content']
            print(f"Application - Index: {index}, Result: {r}")

        # Iterate over result_driver with index
        for index, r in enumerate(result_driver):
            r['sentiment'] = label_mapping.get(r['label'], "unknown")
            r['sentimentName'] = label_mapping_title.get(r['label'], "unknown")
            r['reviewId'] = reviews_categorized_label[index]['id']
            r['reviewsCategory'] = sentiment_categories.get('Driver')
            r['content'] = reviews_categorized_label[index]['content']
            print(f"Driver - Index: {index}, Result: {r}")

        # Iterate over result_attitude with index
        for index, r in enumerate(result_attitude):
            r['sentiment'] = label_mapping.get(r['label'], "unknown")
            r['sentimentName'] = label_mapping_title.get(r['label'], "unknown")
            r['reviewId'] = reviews_categorized_label[index]['id']
            r['reviewsCategory'] = sentiment_categories.get('Attitude')
            r['content'] = reviews_categorized_label[index]['content']
            print(f"Attitude - Index: {index}, Result: {r}")
        return {
            "result_application": result_application,
            "result_driver": result_driver,
            "result_attitude": result_attitude,
            "result_unknown": filter_unknown
        }

    except Exception as e:
        # Catch and return any exception
        return {"error": f"An error occurred: {str(e)}"}