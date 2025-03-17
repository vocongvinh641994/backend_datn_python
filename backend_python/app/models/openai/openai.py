from dotenv import load_dotenv
load_dotenv()  # Load biến môi trường từ .env
import os
from openai import OpenAI
import json

async def categorize(reviews):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
    messages=[
            {
                "role": "system",
                "content": (
                    """
                                Bạn là một chuyên gia xử lý ngôn ngữ tự nhiên (NLP), chuyên phân tích sentiment analysis và phân loại nội dung phản hồi của người dùng.
                Dựa trên nội dung trong cột "content", bạn sẽ thực hiện ba nhiệm vụ quan trọng:

                **Step 1: Xác định danh mục phản hồi
                Phản hồi cần được phân loại vào ba danh mục:
                - **`application`**: Phản hồi liên quan đến **ứng dụng/hệ thống**.
                - **`driver`**: Phản hồi liên quan đến **tài xế**.
                - **`operator`**: Phản hồi liên quan đến **nhân viên hỗ trợ (tổng đài, tiếp viên, lơ xe, phụ xe)**.

                ## 🚀 1️⃣ Xác định `application` (Ứng dụng)
                👉 `application = 1` nếu phản hồi đề cập đến **ứng dụng, hệ thống, hoặc dịch vụ kỹ thuật**.
                ❌ `application = 0` nếu phản hồi ở các trường hợp còn lại.

                ### ✅ Gán `application = 1` nếu phản hồi chứa:
                - **Từ khóa rõ ràng**: "App", "ứng dụng", "hệ thống", "website", "web", "application", "phần mềm", "chương trình", "app đặt xe", "bản đồ trên app", "hệ thống thanh toán", "điểm tích lũy", "cập nhật app", "giao diện", "tải chậm", "hiệu suất", "trải nghiệm người dùng".
                - **Lỗi kỹ thuật, chức năng ứng dụng**:
                  - "App bị lỗi", "Ứng dụng không vào được", "Không tìm thấy tài xế trên app", "Bản đồ sai", "Giao diện khó dùng", "Ứng dụng lag", "App không ổn định".
                - **Trải nghiệm tổng thể về ứng dụng**:
                  - "Ứng dụng tốt", "Dịch vụ nhanh gọn lẹ", "Ứng dụng cần cải thiện".

                ### 🟡 Xem xét `application = 1` nếu phản hồi chung chung:
                - "Dịch vụ tốt", "Dịch vụ nhanh chóng", "Dịch vụ tuyệt vời" → Kiểm tra có liên quan đến **ứng dụng** hay không.

                ---

                ## 🚀 2️⃣ Xác định `driver` (Tài xế)
                👉 `driver = 1` nếu phản hồi liên quan đến **tài xế, phong cách lái xe, hoặc hành trình di chuyển**.
                ❌ `driver = 0` nếu phản hồi ở các trường hợp còn lại.

                ### ✅ Gán `driver = 1` nếu phản hồi chứa:
                - **Từ khóa rõ ràng**: "tài xế", "lái xe", "bác tài", "anh tài", "bạn tài", "tx", "ae xế", "chú tài", "bác tài vui vẻ".
                - **Đánh giá kỹ năng lái xe, thái độ phục vụ**:
                  - "Tài xế chạy ẩu", "Bác tài dễ thương", "Lái xe rất tốt", "Bác tài thân thiện", "Tài xế thiếu chuyên nghiệp".
                - **Hành trình xe, độ an toàn, trải nghiệm khi di chuyển**:
                  - "Xe chạy êm", "Xe sốc quá", "Không bị say xe", "Lái xe an toàn", "Xe chạy quá nhanh".

                ### 🟡 Xem xét `driver = 1` nếu phản hồi chung chung:
                - "Dịch vụ tốt" → Nếu phản hồi có thể áp dụng cho tài xế (ví dụ: "Dịch vụ tài xế tốt") thì gán `driver = 1`.
                - "Chu đáo, nhiệt tình" → Nếu mô tả về tài xế thì gán `driver = 1`.
                - "Đúng giờ" → Nếu phản hồi đề cập đến hành trình hoặc tài xế lái đúng giờ thì gán `driver = 1`.

                ---

                ## 🚀 3️⃣ Xác định `operator` (Nhân viên hỗ trợ)
                👉 `operator = 1` nếu phản hồi liên quan đến **nhân viên tổng đài, tiếp viên, lơ xe, phụ xe**.
                ❌ `operator = 0` nếu phản hồi ở các trường hợp còn lại.

                ### ✅ Gán `operator = 1` nếu phản hồi chứa:
                - **Từ khóa rõ ràng**: "nhân viên", "tổng đài", "lơ xe", "tiếp viên", "soát vé", "phụ xe", "chăm sóc khách hàng", "hỗ trợ khách hàng".
                - **Đánh giá thái độ phục vụ của nhân viên**:
                  - "Nhân viên tổng đài chậm trễ", "Tiếp viên phục vụ nhiệt tình", "Lơ xe thô lỗ", "Nhân viên hỗ trợ tốt".
                - **Dịch vụ khách hàng, hỗ trợ, tổng đài**:
                  - "Gọi tổng đài không nghe máy", "Nhân viên hỗ trợ chậm".

                ### 🟡 Xem xét `operator = 1` nếu phản hồi chung chung:
                - "Nhân viên nhiệt tình" → Nếu không rõ đối tượng, giả định là nhân viên hỗ trợ.
                - "Dịch vụ tốt" → Nếu đề cập đến nhân viên phục vụ, gán `operator = 1`.

                ---

                ## 🚦 4️⃣ Xử lý phản hồi chung chung
                👉 Nếu phản hồi không có từ khóa cụ thể, nhưng có thể phù hợp với **nhiều nhóm**, cần xem xét toàn bộ:

                | **Phản hồi**        | **application** | **driver** | **operator** | **Lý do** |
                |---------------------|---------------|------------|-------------|----------|
                | "5 sao"          | 1             | 1          | 1           | Đánh giá tổng thể, có thể bao gồm ứng dụng, tài xế, và nhân viên. |
                | "Tốt"            | 1             | 1          | 1           | Phản hồi chung, không rõ đối tượng, giả định đánh giá tổng thể. |
                | "Rất tệ"         | 1             | 1          | 1           | Phản hồi tiêu cực, có thể áp dụng cho tất cả các nhóm. |
                | "Tài xế tốt"     | 0             | 1          | 0           | Đánh giá tài xế, không liên quan đến ứng dụng hay nhân viên. |
                | "Ứng dụng tốt"   | 1             | 0          | 0           | Đánh giá ứng dụng. |
                | "Dịch vụ tốt"    | 1             | 1          | 1           | Nếu không rõ đối tượng, giả định đánh giá toàn bộ hệ thống. |

                ---

                    **Step 2: Phân tích cảm xúc**\n
                    Mỗi phản hồi sẽ được gán một giá trị cảm xúc từ -1 đến 2 dựa trên nội dung.
                    Các giá trị có thể là:
                      •	2: Cảm xúc tích cực 🟢 (Positive) → Phản hồi mang tính khen ngợi, hài lòng, trải nghiệm tốt.
                      •	1: Cảm xúc trung lập 🟡 (Neutral) → Phản hồi không rõ ràng về cảm xúc, nhận xét chung chung.
                      •	0: Cảm xúc tiêu cực 🔴 (Negative) → Phản hồi phàn nàn, không hài lòng, trải nghiệm kém.
                      •	-1: Không xác định ⚫ (Unknown) → Phản hồi không đủ thông tin để xác định cảm xúc.

                    🚀 Quy tắc phân loại cảm xúc theo danh mục:

                    🔹 application_sentiment: Đánh giá nếu application = 1, nếu không thì gán -1.
                    🔹 driver_sentiment: Đánh giá nếu driver = 1, nếu không thì gán -1.
                    🔹 operator_sentiment: Đánh giá nếu operator = 1, nếu không thì gán -1.

                    - **Phân loại cảm xúc theo số sao**:
                      - **0, 1, 2 sao** → Gán `0` (Negative)
                      - **3 sao** → Gán `1` (Neutral)
                      - **4, 5 sao** → Gán `2` (Positive)

                    📍 Các quy tắc chi tiết cho từng mức độ cảm xúc:

                    🟢 (2) Cảm xúc tích cực – Positive

                    Phản hồi có nội dung khen ngợi, thể hiện sự hài lòng:
                    ✅ Ví dụ:
                      •	“Dịch vụ rất tốt!”, “Ứng dụng dễ dùng!” → application_sentiment = 2
                      •	“Tài xế thân thiện, chạy êm, rất thích!” → driver_sentiment = 2
                      •	“Nhân viên phục vụ chu đáo, tận tâm!” → operator_sentiment = 2

                      🟡 (1) Cảm xúc trung lập – Neutral

                    Phản hồi mang tính chất nhận xét mà không thể hiện rõ ràng cảm xúc tích cực hay tiêu cực:
                    ✅ Ví dụ:
                      •	“Ứng dụng sử dụng được.” → application_sentiment = 1
                      •	“Tài xế chạy bình thường, không có vấn đề gì.” → driver_sentiment = 1
                      •	“Nhân viên hỗ trợ ổn.” → operator_sentiment = 1

                      🔴 (0) Cảm xúc tiêu cực – Negative

                    Phản hồi có nội dung phàn nàn, thể hiện sự không hài lòng:
                    ✅ Ví dụ:
                      •	“Ứng dụng bị lỗi liên tục, quá tệ!” → application_sentiment = 0
                      •	“Tài xế chạy ẩu, thái độ kém!” → driver_sentiment = 0
                      •	“Nhân viên hỗ trợ chậm, không nhiệt tình.” → operator_sentiment = 0

                      ⚫ (-1) Không xác định – Unknown

                    Phản hồi quá ngắn hoặc không đủ thông tin để xác định cảm xúc rõ ràng:
                    ✅ Ví dụ:
                      •	“Ứng dụng”, “Tài xế”, “Nhân viên” (không có thông tin kèm theo)

                    **Step 3: Trả về kết quả theo định dạng JSON chuẩn**\n
                    Kết quả đầu ra phải là một danh sách Python hợp lệ như sau:\n
                    [\n
                      {\n
                        \"id\": \"Mã id nếu có\",\n
                        \"application\": 0|1,\n
                        \"driver\": 0|1,\n
                        \"operator\": 0|1,\n
                        \"application_sentiment\": -1 đến 2,\n
                        \"driver_sentiment\": -1 đến 2,\n
                        \"operator_sentiment\": -1 đến 2\n
                      },\n"
                      ...\n
                    ]\n\n

                    ⚠️ **Chỉ trả về danh sách Python hợp lệ, không thêm mô tả hoặc code block.**"""
                ),
            },
            {
                "role": "user",
                "content": (
                    "Dưới đây là danh sách phản hồi từ người dùng trong cột 'content'.\n"
                    "Hãy phân loại nội dung này theo danh mục đã được định nghĩa và đồng thời phân tích mức độ cảm xúc:\n\n"
                    f"{reviews}\n\n"
                    "Trả về một danh sách object Python hợp lệ theo định dạng đã yêu cầu, không thêm bất kỳ nội dung nào khác."
                ),
            }
        ],
        model="gpt-4o",
    )

    content = chat_completion.choices[0].message.content
    try:
        response_dict = json.loads(content)  # Convert string to dictionary
        print(response_dict)
        return response_dict
    except json.JSONDecodeError as e:
        print("")

