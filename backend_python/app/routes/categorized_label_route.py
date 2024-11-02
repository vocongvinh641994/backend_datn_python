from fastapi import APIRouter, Request
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

router = APIRouter()

path_finetuned = "app/models/phobert_models/phobert_categorizedLabel/2024_09_28/"
model = AutoModelForSequenceClassification.from_pretrained(path_finetuned)
tokenizer = AutoTokenizer.from_pretrained(path_finetuned)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU if available
model.eval()  # Set model to evaluation mode

# Enable half-precision if using a GPU
if torch.cuda.is_available():
    model.half()

@router.post("/categorized_label/")
async def classify_phobert(request: Request):
    data = await request.json()
    reviews_param = data.get("reviews", [])
    result = await categorizedLabel(reviews_param)
    return result

async def categorizedLabel(reviews_param):
    try:
        reviews = [review["content"] for review in reviews_param]
        if not reviews or not isinstance(reviews, list):
            return {"error": "Invalid or missing 'reviews' field. Please provide a list of reviews."}

        # Tokenize inputs and move them to the correct device (GPU/CPU)
        inputs = tokenizer(
            reviews,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        # Perform inference with no gradients
        with torch.no_grad():
            outputs = model(**inputs)

        # Get raw logits and apply sigmoid activation
        logits = outputs.logits
        predictions = torch.sigmoid(logits)

        # Convert predictions to binary (0 or 1)
        predicted_labels = (predictions >= 0.3).int()

        # Label names
        label_names = ['label_application', 'label_attitude', 'label_driver']

        for review_idx, review_obj in enumerate(reviews_param):
            for i, label in enumerate(predicted_labels[review_idx]):
                review_obj[label_names[i]] = int(label)

        return reviews_param

    except Exception as e:
        return {"error": str(e)}