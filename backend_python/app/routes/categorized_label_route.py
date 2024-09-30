from fastapi import APIRouter, Request
from torch.export import export
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

router = APIRouter()

# Path to your finetuned model directory
path_finetuned = "app/models/phobert_models/phobert_categorizedLabel/2024_09_28/"
# Load the model and tokenizer from the fine-tuned path
model = AutoModelForSequenceClassification.from_pretrained(path_finetuned)
tokenizer = AutoTokenizer.from_pretrained(path_finetuned)

model.eval()  # Set the model to evaluation mode
@router.post("/categorized_label/")
async def classify_phobert(request: Request):
    data = await request.json()
    reviews_param = data.get("reviews", [])
    # Use categorizedLabel function
    result = await categorizedLabel(reviews_param)
    return result

async def categorizedLabel(reviews_param):
    try:
        reviews = [review["content"] for review in reviews_param]  # Validate 'reviews' field
        if not reviews or not isinstance(reviews, list):
            return {"error": "Invalid or missing 'reviews' field. Please provide a list of reviews."}
        # Tokenize the list of reviews
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

        # Convert predictions to binary (0 or 1) with a threshold of 0.5
        predicted_labels = (predictions > 0.5).int()

        # Label names for interpretation
        label_names = ['label_application', 'label_attitude', 'label_driver']
        # Display predictions for each review
        for review_idx, review_obj in enumerate(reviews_param):
            for i, label in enumerate(predicted_labels[review_idx]):
                review_obj[label_names[i]] = int(label)
        return reviews_param

    except Exception as e:
        # Catch and return any exception
        return []
