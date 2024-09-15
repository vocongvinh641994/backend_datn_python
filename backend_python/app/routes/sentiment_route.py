from fastapi import APIRouter, Request
from transformers import pipeline, AutoTokenizer
import torch
from fastapi.concurrency import run_in_threadpool

router = APIRouter()

# Path to your finetuned model directory
path_finetuned_phobert_attitude = "app/models/phobert_models/phobert_attitude/checkpoint_20240812_epoch1"
path_finetuned_phobert_driver = "app/models/phobert_models/phobert_driver/checkpoint_20240819_epoch5"
path_finetuned_phobert_application = "app/models/phobert_models/phobert_application/checkpoint_20240812_epoch1"
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Define label mapping
label_mapping = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# Keywords
keywords_application = ["App", "app", "ứng dụng"]
keywords_attitude = ["thân thiện", "nhiệt tình", "chu đáo", "lịch sự", "tận tâm", "chuyên nghiệp", "hỗ trợ tốt",
                     "nhiệt tình hỗ trợ", "vui vẻ", "dễ chịu", "tận tình", "tốt bụng", "thô lỗ", "thiếu chuyên nghiệp",
                     "thiếu lịch sự", "khó chịu", "kém lịch sự", "bực mình", "quá đáng", "thiếu tôn trọng", "chậm chạp",
                     "thiếu nhiệt tình", "khó ưa", "lơ là", "không hài lòng", "không chu đáo", "không nhiệt tình",
                     "không hỗ trợ", "phục vụ kém", "phục vụ tệ", "dịch vụ kém", "dịch vụ tệ", "phản hồi chậm"]
keywords_driver = ["Tài xế", "Nhân viên", "Thái độ", "Phục vụ", "Chăm sóc khách hàng", "Hỗ trợ", "Thân thiện",
                   "Nhiệt tình", "Lịch sự", "Chu đáo", "Tận tâm", "Kỹ năng giao tiếp", "Lắng nghe", "Thấu hiểu",
                   "Giải quyết vấn đề", "Trách nhiệm", "Tôn trọng", "Chuyên nghiệp", "Hợp tác"]

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
        filtered_reviews_application = reviews
        # filtered_reviews_application = [review for review in reviews if
        #                                 any(keyword in review['content'] for keyword in keywords_application)]
        filtered_reviews_driver = [review for review in reviews if
                                   any(keyword in review['content'] for keyword in keywords_driver)]
        filtered_reviews_attitude = [review for review in reviews if
                                     any(keyword in review['content'] for keyword in keywords_attitude)]

        # Extract texts for sentiment analysis
        texts_application = [review['content'] for review in filtered_reviews_application]
        texts_driver = [review['content'] for review in filtered_reviews_driver]
        texts_attitude = [review['content'] for review in filtered_reviews_attitude]

        # Create sentiment-analysis pipelines using GPU (if available)
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
        for r in result_application:
            r['label'] = label_mapping.get(r['label'], "unknown")
        for r in result_driver:
            r['label'] = label_mapping.get(r['label'], "unknown")
        for r in result_attitude:
            r['label'] = label_mapping.get(r['label'], "unknown")

        return {
            "result_application": result_application,
            "result_driver": result_driver,
            "result_attitude": result_attitude
        }

    except Exception as e:
        # Catch and return any exception
        return {"error": f"An error occurred: {str(e)}"}