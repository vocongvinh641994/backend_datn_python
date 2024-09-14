from fastapi import APIRouter, Request
from transformers import pipeline, AutoTokenizer

router = APIRouter()

# Path to your finetuned model directory
path_finetuned_phobert = "app/models/phobert_models/phobert_driver/checkpoint"
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Define label mapping
label_mapping = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

@router.post("/classify_phobert/")
async def classify_phobert(request: Request):
    try:
        # Parse the incoming JSON request
        data = await request.json()
        texts = data.get("texts", [])

        # Check if 'texts' is present and is a list
        if not texts or not isinstance(texts, list):
            return {"error": "Invalid or missing 'texts' field. Please provide a list of texts."}

        # Create a sentiment-analysis pipeline using your finetuned model and tokenizer
        classifier = pipeline("sentiment-analysis", model=path_finetuned_phobert, tokenizer=tokenizer)

        # Classify the texts using the pipeline
        result = classifier(texts)

        # Map the labels to meaningful sentiment values (positive, neutral, negative)
        for r in result:
            r['label'] = label_mapping.get(r['label'], "unknown")

        return {"result": result}

    except Exception as e:
        # Catch and return any exception
        return {"error": f"An error occurred: {str(e)}"}