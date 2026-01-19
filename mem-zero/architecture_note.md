# ĐẶC TẢ KIẾN TRÚC MEMORY MEM0 (PRODUCTION-READY)

## 1. Mô hình Dữ liệu (Data Model)

Trước khi vào luồng xử lý, bạn cần định nghĩa cấu trúc dữ liệu lưu trong Vector Database (như Qdrant/Pinecone) và SQL/NoSQL (lưu Summary).

### A. Memory Item (Lưu trong Vector DB)
Mỗi "ký ức" là một sự thật (fact) đơn lẻ, không phải đoạn văn dài.
```json
{
  "id": "uuid-v4",
  "text": "User thích ăn pizza chay nhưng ghét hành tây",  // Nội dung dùng để tạo embedding
  "vector": [0.12, -0.5, ...], // Embedding (Paper dùng text-embedding-small-3)
  "metadata": {
    "user_id": "user_123",
    "agent_id": "agent_sales", // Quan trọng cho multi-agent
    "created_at": "2023-10-27T10:00:00Z",
    "updated_at": "2023-10-27T10:05:00Z",
    "topic": "food_preference" // Optional: phân loại để filter nhanh hơn
  }
}
```

### B. Conversation Context (Lưu trong DB thường/Redis)
```json
{
  "session_id": "sess_abc",
  "global_summary": "Người dùng đang thảo luận về sở thích ăn uống và kế hoạch du lịch...",
  "recent_messages": [ ... ] // List 10 tin nhắn gần nhất (Rolling window)
}
```

---

## 2. Pipeline Xử lý Chi tiết (The Mem0 Workflow)

Quy trình chia làm 2 luồng: **Luồng Nóng (Hot Path - Trả lời User)** và **Luồng Lạnh (Cold Path - Cập nhật Memory)**. Để tối ưu Latency cho Production, 2 luồng này nên chạy song song hoặc bất đồng bộ.

### Giai đoạn 1: Truy xuất & Trả lời (Retrieval & Generation)
*Mục tiêu: Trả lời User nhanh nhất có thể (Target: < 1.5s).*

1.  **Input:** User Query ($Q$).
2.  **Vector Search:**
    *   Embedding $Q$.
    *   Search trong Vector DB với filter `user_id`.
    *   Lấy Top-K (Paper dùng **k=10**).
3.  **Generation:**
    *   Đưa các memories tìm được vào System Prompt của Agent.
    *   **Prompt (Từ Paper - Appendix A & Results Generation):**
        > "You are an intelligent memory assistant...
        > Context: You have access to memories...
        > Instructions:
        > 1. Prioritize most recent memory if contradictory.
        > 2. Convert relative time (last year) to specific dates using timestamp..."
        *(Xem chi tiết prompt đầy đủ ở mục 3 bên dưới)*.
4.  **Output:** Trả về câu trả lời cho User.

### Giai đoạn 2: Trích xuất & Cập nhật (Extraction & Update)
*Mục tiêu: Ghi nhớ thông tin mới, loại bỏ tin cũ/sai. Chạy ngay sau khi có cặp (Q, A).*

1.  **Input:** Cặp tin nhắn mới nhất ($m_{t-1}, m_t$) + Global Summary ($S$) + 10 tin nhắn gần nhất.
2.  **Extraction (LLM Call 1):**
    *   Trích xuất các "facts" từ cuộc hội thoại.
    *   **Prompt Logic:** "Dựa trên tóm tắt $S$ và ngữ cảnh hội thoại, hãy trích xuất các sự thật quan trọng từ tin nhắn mới nhất."
    *   **Output:** List $\Omega = [f_1, f_2, ...]$.
3.  **Update Loop (Vòng lặp quan trọng nhất):**
    Với mỗi fact $f_i$ trong list $\Omega$:
    *   **Bước 3a: Search Tương đồng:** Query Vector DB tìm top-10 memories cũ giống $f_i$ nhất (gọi là $M_{existing}$).
    *   **Bước 3b: Function Calling (LLM Call 2 - Algorithm 1):**
        *   Gửi $f_i$ và $M_{existing}$ cho LLM.
        *   LLM quyết định chọn function nào:
            *   `ADD(f_i)`: Nếu chưa có thông tin này.
            *   `UPDATE(id_old, f_new)`: Nếu thông tin mới chi tiết hơn.
            *   `DELETE(id_old)`: **Nếu thông tin mới mâu thuẫn** (VD: User bảo "Tôi hết thích hành rồi").
            *   `NOOP`: Không làm gì.
    *   **Bước 3c: Execute:** Thực thi lệnh lên Vector DB.

### Giai đoạn 3: Tóm tắt Bất đồng bộ (Async Summarization)
*   **Trigger:** Chạy background job sau mỗi $N$ lượt hội thoại.
*   **Action:** Cập nhật `Global Summary` để phục vụ cho Giai đoạn 2 lần sau.

---

## 3. Chi tiết các Prompt (Trích xuất từ Paper)

Dưới đây là các prompt quan trọng để bạn cấu hình cho LLM.

### A. Prompt cho Generation (Trả lời câu hỏi)
*Nguồn: Appendix của Paper. Dùng khi Agent trả lời User.*

```text
SYSTEM PROMPT:
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information...

# INSTRUCTIONS:
1. Carefully analyze all provided memories.
2. Pay special attention to the timestamps to determine the answer.
3. If the memories contain contradictory information, prioritize the most recent memory.
4. If there is a question about time references (like "last year"), calculate the actual date based on the memory timestamp.
5. Focus only on the content of the memories.

# MEMORIES PROVIDED:
{retrieved_memories_list}

# USER QUESTION:
{user_question}
```

### B. Prompt Logic cho Extraction (Trích xuất sự thật)
*Paper mô tả logic ở section 2.1, không in nguyên văn, nhưng đây là cấu trúc tái tạo chính xác logic đó:*

```text
SYSTEM PROMPT:
You are a memory manager AI. Your goal is to extract salient facts from the current conversation turn to update the Knowledge Base.

# INPUT CONTEXT:
Global Summary: {S}
Recent Messages: {last_10_messages}

# CURRENT INTERACTION:
User: {m_t-1}
Assistant: {m_t}

# TASK:
Extract distinct, concise facts (memories) from the "Current Interaction".
- Ignore trivial chit-chat.
- Focus on user preferences, specific events, established facts.
- Output a JSON list of strings. Example: ["User is vegetarian", "User lives in Hanoi"].
```

### C. Prompt Logic cho Update (Function Calling)
*Đây là "trái tim" của Mem0. Bạn nên dùng tính năng Function Calling/Tools của OpenAI/LangChain.*

```text
Tool Definition (Function Schema):
{
  "name": "manage_memory",
  "description": "Decide how to update the memory database based on new fact and existing memories.",
  "parameters": {
    "type": "object",
    "properties": {
      "operation": {
        "type": "string",
        "enum": ["ADD", "UPDATE", "DELETE", "NOOP"],
        "description": "The action to perform."
      },
      "target_memory_id": {
        "type": "string",
        "description": "The ID of the existing memory to UPDATE or DELETE. Null if ADD."
      },
      "new_content": {
        "type": "string",
        "description": "The new content for ADD or UPDATE."
      }
    },
    "required": ["operation"]
  }
}

SYSTEM PROMPT for Tool Call:
You are a consistent knowledge base maintainer.
New Fact: "{new_fact}"
Existing Similar Memories:
[
  {"id": "1", "content": "User likes chicken", "score": 0.85},
  {"id": "2", "content": "User hates onions", "score": 0.3}
]

Task: Determine the relationship between the New Fact and Existing Memories.
- If New Fact contradicts an Existing Memory -> DELETE the Existing Memory ID and ADD the new one (or UPDATE).
- If New Fact adds detail to Existing Memory -> UPDATE Existing Memory ID.
- If New Fact is completely new -> ADD.
- If New Fact is already covered -> NOOP.
```

---

## 4. Benchmarks & KPIs (Để so sánh và tối ưu)

Đây là các con số bạn cần nhắm tới (Target) khi code lại để đảm bảo hiệu năng tương đương bản gốc.

### A. Độ trễ (Latency Targets) - *Đo trên GPT-4o-mini*

| Thành phần | Target (p50) | Target (p95 - Worst Case) | Giải thích |
| :--- | :--- | :--- | :--- |
| **Search Latency** | **~0.15s** | **< 0.20s** | Thời gian query Vector DB. Nếu > 0.5s là code chưa tối ưu index. |
| **Total Response Time** | **~0.70s** | **< 1.50s** | Tổng thời gian từ lúc User hỏi đến lúc nhận câu trả lời (Retrieval + Gen). |
| **Full Pipeline Update** | N/A | **< 60s** | Thời gian để cập nhật ký ức (Extraction + Update). Mem0 làm điều này cực nhanh so với Zep. |

### B. Hiệu quả Token (Cost Efficiency)

| Chỉ số | Mem0 Target | Full-Context (RAG) | Ghi chú |
| :--- | :--- | :--- | :--- |
| **Tokens lưu trữ** | **~7k - 14k** | Tăng vô hạn theo hội thoại | Mem0 nén hội thoại thành facts nên token rất thấp. |
| **Tokens input/query** | **Thấp** | ~26k (cho hội thoại dài) | Tiết kiệm ~90% chi phí API. |

---

## 5. Lời khuyên cho Multi-Agent Production

Khi áp dụng kiến trúc này cho hệ thống Multi-Agent, bạn cần lưu ý 3 điểm sau để không bị vỡ trận:

1.  **Namespace trong Vector DB:**
    *   Cần phân chia rõ ràng. Memory của `Agent A` có được `Agent B` nhìn thấy không?
    *   *Design:* Thêm field `scope: "shared" | "private"` vào metadata của Memory. Khi search vector, luôn filter theo `scope`.
2.  **Concurrency Locking:**
    *   Nếu 2 agent cùng nói chuyện với 1 user và cùng update memory, sẽ xảy ra Race Condition.
    *   *Giải pháp:* Dùng Queue (Redis Queue) cho luồng Update Phase. Các request cập nhật memory sẽ được xếp hàng và xử lý tuần tự (Serial execution) cho từng User ID.
3.  **Graph Memory (Mem0g) - Should or Should Not?**
    *   Paper chỉ ra rằng **Mem0 (Vector thuần)** tốt hơn cho các truy vấn đơn giản và nhanh hơn (Latency thấp hơn).
    *   **Mem0g (Graph)** chỉ tốt hơn ở các câu hỏi **Temporal** (thời gian) và **Open-domain**.
    *   *Khuyên dùng:* Bắt đầu với **Vector thuần** (Mem0 base) trước. Chỉ implement Graph nếu bạn cần agent suy luận phức tạp về mối quan hệ (VD: A là bố của B, B sống ở C => A có con sống ở C).

Đây là bản thiết kế hoàn chỉnh. Bạn có thể bắt tay vào code phần `Vector Search` và `Function Calling` ngay vì đó là 2 phần quan trọng nhất tạo nên sự khác biệt của Mem0.