from dotenv import load_dotenv
load_dotenv()  # Load biến môi trường từ .env
import os
from openai import OpenAI
import json

async def categorize(reviews):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),  # This is the default and can be omitted
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là một chuyên gia về xử lý ngôn ngữ tự nhiên, chuyên phân tích sentiment analysis "
                    "và phân loại nội dung phản hồi của người dùng. "
                    "Dựa trên nội dung trong cột 'content', bạn sẽ thực hiện hai nhiệm vụ quan trọng:\n\n"

                    "1️⃣ **Phân loại nội dung phản hồi** thành một trong các danh mục sau:\n"
                    "- 0: 'application' - nếu phản hồi liên quan đến ứng dụng.\n"
                    "- 1: 'driver' - nếu phản hồi liên quan đến tài xế.\n"
                    "- 2: 'operator' - nếu phản hồi liên quan đến nhân viên tổng đài hoặc nhân viên phục vụ.\n"
                    "- 3: 'application_driver' - nếu phản hồi đề cập đến cả ứng dụng và tài xế.\n"
                    "- 4: 'application_operator' - nếu phản hồi đề cập đến cả ứng dụng và nhân viên tổng đài hoặc tiếp viên.\n"
                    "- 5: 'driver_operator' - nếu phản hồi đề cập đến cả tài xế và nhân viên tổng đài hoặc tiếp viên.\n"
                    "- 6: 'application_driver_operator' - nếu phản hồi đề cập đến cả ứng dụng, tài xế và nhân viên tổng đài hoặc tiếp viên.\n"
                    "- 7: 'unknown' - nếu phản hồi không thuộc bất kỳ danh mục nào ở trên.\n\n"

                    "2️⃣ **Phân tích cảm xúc (sentiment analysis)** của phản hồi theo các mức độ sau:\n"
                    "- 0: 'Negative' - nếu phản hồi mang tính tiêu cực.\n"
                    "- 1: 'Neutral' - nếu phản hồi mang tính trung lập.\n"
                    "- 2: 'Positive' - nếu phản hồi mang tính tích cực.\n"
                    "- 3: 'Unknown' - nếu không thể xác định được cảm xúc.\n\n"
                     "⚠️ **Nếu là tiếng việt thì phân biệt được các phản hồi không có dấu.**"

                    "📌 **Yêu cầu:**\n"
                    "- Đọc danh sách phản hồi trong cột 'content'.\n"
                    "- Trả về kết quả dưới dạng một **Python object thuần túy** (từ điển Python), không phải chuỗi JSON hay code block.\n"
                    "- Đảm bảo mỗi phần tử trong `category` tương ứng với một phần tử trong `sentiment`.\n\n"

                    "📤 **Kết quả mong muốn (dưới dạng object Python hợp lệ):**\n"
                    "{\n"
                    "  \"category\": [danh sách các giá trị từ 0-7 tương ứng với phân loại],\n"
                    "  \"sentiment\": [danh sách các giá trị từ 0-3 tương ứng với mức độ cảm xúc]\n"
                    "}\n\n"

                    "⚠️ **Chỉ trả về đối tượng Python hợp lệ, không thêm bất kỳ văn bản nào khác.**"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Dưới đây là danh sách phản hồi từ người dùng trong cột 'content'. "
                    f"Hãy phân loại nội dung này theo danh mục đã được định nghĩa và đồng thời phân tích mức độ cảm xúc:\n\n"
                    f"{reviews}\n\n"
                    f"Hãy trả về một object Python hợp lệ với hai danh sách: `category` và `sentiment`, không thêm bất kỳ nội dung nào khác."
                ),
            },
        ],
        model="gpt-4o",
    )

    content = chat_completion.choices[0].message.content
    print("Raw response:", repr(content))  # Shows the exact format of the response
    print(content)  # Expected output: <class 'str'>
    import json
    try:
        response_dict = json.loads(content)  # Convert string to dictionary
        category_values = response_dict["category"]  # Now you can access category
        sentiment_values = response_dict["sentiment"]  # Now you can access category
        print("Category:", category_values)  # Output: [7, 2]
        print("Sentiment:", sentiment_values)  # Output: [7, 2]
        return response_dict
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)

