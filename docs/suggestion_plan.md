# Chiến lược Tối ưu hóa Hệ thống Multi-Agent: Kiến trúc, Memory & Slot Filling

Tài liệu này mô tả kiến trúc tối ưu cho hệ thống multi-agent với các tiêu chí: **tốc độ phản hồi nhanh**, **tiết kiệm token**, và **chính sách lưu trữ chặt chẽ** (chỉ lưu khi hoàn tất slot-filling).

---

## 1. Tóm tắt mục tiêu thiết kế

*   **Low Latency:** Giữ độ trễ thấp cho các quyết định của agent.
*   **Token Efficiency:** Chỉ đưa vào prompt những thông tin thật sự cần thiết cho inference (suy luận).
*   **Persist Policy:** Chỉ lưu trữ dài hạn (persist) khi episode hoàn thành (user confirm đủ các trường). Trước đó sử dụng bộ nhớ tạm (ephemeral).
*   **Multi-agent Support:** Chia memory thành **Local** (agent-scoped) và **Shared** (global/consensus).
*   **Compression:** Có cơ chế nén/tóm tắt lịch sử để giảm tải token.

---

## 2. Kiến trúc 3-Tier Memory

### Tier 1: Working Memory (In-Prompt, Ephemeral)
*   **Nội dung:** Last N turns (thô) + Current Slot Status (dạng bảng rút gọn).
*   **Đặc điểm:** Lưu ở runtime, không persist vào DB. Kích thước cố định (ví dụ: 1–2 lượt hội thoại gần nhất).
*   **Mục tiêu:** Cung cấp ngữ cảnh tức thì cho LLM xử lý câu hiện tại.

### Tier 2: Short-Term Memory (Session / Slot Pool)
*   **Công nghệ:** Redis (Hashes, TTL) hoặc In-memory store.
*   **Nội dung:** Slot pool (giá trị + trạng thái), temporary confirmations, action logs nhỏ.
*   **TTL:** Ngắn (ví dụ: 30–60 phút).
*   **Mục tiêu:** Dùng để resume session, xử lý multi-turn slot filling, rollback hội thoại. Nếu session idle quá TTL, dữ liệu tự động bị xóa (abandoned).

### Tier 3: Long-Term Memory (Persistent, Retrievable)
*   **Công nghệ:** Vector DB (Pinecone/Weaviate/Chroma) + RDB/Document DB (Postgres/SQLite) cho metadata.
*   **Nội dung:** Episodes đã hoàn chỉnh (confirmed), user profiles summary (compact), templates/consensus rules.
*   **Index:** Embeddings + Metadata (domain, category, timestamp, importance).
*   **Mục tiêu:** Lưu trữ kiến thức lâu dài sau khi đã được xác thực.

---

## 3. Slot Filling & Persist Policy

### Cấu trúc Slot Pool (Redis per session)
```json
{
  "session_id": "session_12345",
  "slots": {
    "destination": {
      "value": "Hanoi",
      "confirmed": false,
      "source": "user_text"
    },
    "date": {
      "value": "2026-01-20",
      "confirmed": true,
      "source": "agent_prompt"
    }
  },
  "last_updated": "2026-01-13T10:00:00Z",
  "state": "filling" 
  // Các trạng thái: "filling", "ready_to_persist", "abandoned"
}
```

### Luồng xử lý (Flow)
1.  **User Message:** Message tới -> Chạy Regex/NLU nhanh để populate slot pool.
2.  **Check Missing:** Nếu thiếu slot -> Agent tạo câu hỏi làm rõ (clarifying question) tối giản -> Chỉ ghi vào slot pool khi user trả lời.
3.  **Persist Condition:** Khi tất cả `required_slots` có `confirmed == true` -> Đánh dấu `state = "ready_to_persist"`.
4.  **Save:** Chỉ tại thời điểm này mới ghi Episode vào Long-Term DB và gọi API.
5.  **Cleanup:** Nếu session idle > TTL -> Đánh dấu `abandoned` và xóa khỏi Short-term memory.

---

## 4. Pipeline Trích xuất dữ liệu (Extraction Pipeline)

Để tiết kiệm token và tăng tốc độ:

*   **Stage 0 — Fast Pattern Matching (Regex/Rule):** Xử lý email, số điện thoại, ngày tháng, con số. Cực nhanh, chi phí 0 token.
*   **Stage 1 — Lightweight NER:** Sử dụng các model nhỏ (BERT-small / Distilled) để phân biệt các thực thể nhập nhằng.
*   **Stage 2 — LLM (Chỉ khi cần thiết):** Dùng prompt tập trung cao độ, chỉ truyền `Slot Table` + `New User Text` (tránh truyền full chat history).

---

## 5. Chiến lược Prompting & Token Budget

### Prompt Template (Slot-driven)
Chỉ bao gồm:
1.  Task Instruction (1-2 dòng).
2.  Current Slot Table (Compact: `name: value`).
3.  New User Utterance.
4.  Requested Action.

### Hierarchical Summarization
Khi episode được persist, lưu 2 dạng:
*   **Short Summary:** ~1-3 câu (dùng để đưa vào prompt nhanh).
*   **Embedding:** Dùng cho Vector Search.

### Dynamic Budget Allocation
Phân bổ token cho prompt:
*   **40%:** System instruction + Current user question.
*   **40%:** Slot table + Short summary (context).
*   **20%:** Spare (dự phòng).

---

## 6. Chiến lược Truy xuất (Retrieval - Hybrid)

Kết hợp để giảm token và tăng độ chính xác:

1.  **Sparse Retrieval (Lọc thô):** Dùng BM25 hoặc SQL Filters trên metadata để lọc nhanh danh sách ứng viên (Chi phí thấp).
2.  **Dense Retrieval (Lọc tinh):** Dùng Vector similarity trên danh sách đã lọc.
3.  **Scoring Function:**
    ```python
    Score = 0.45 * semantic_sim + 0.30 * bm25_score + 0.15 * recency + 0.10 * importance
    ```
4.  **Output:** Chỉ attach **Top-k** (k nhỏ, vd: 1-2) *short summaries* vào prompt, không đưa full episodes.

---

## 7. Multi-Agent Coordination & Shared Memory

*   **Consensus Memory:** Sử dụng RDB nhỏ hoặc Graph DB (Neo4j) lưu:
    *   Verified flows (vd: "booking_flow_v1").
    *   Shared policies.
    *   Dependency states.
*   **Interaction Graph:** Lưu references (`episode_id`) trong metadata của Vector DB; các agents có thể subscribe thay đổi.
*   **Locking:** Khi Agent A update shared-state, dùng **Redis Lock** để tránh race condition.

---

## 8. Chính sách Nén & Quên (Compression / Forgetting)

*   **Episodic Condensation:** Sau X ngày hoặc khi số lượng episode lớn -> Chạy job tóm tắt tự động -> Nén thành 1-2 câu digest -> Lưu digest + embedding -> Xóa raw text.
*   **Importance Tagging:** User hoặc hệ thống đánh dấu "Quan trọng" -> Giữ lại lâu hơn (High retention).
*   **Forgetting Window:**
    *   30 ngày: Giữ raw text.
    *   30 ngày - 1 năm: Giữ dạng nén (compressed).
    *   > 1 năm: Xóa hoặc Archive.

---

## 9. Bảo mật & Quyền riêng tư

*   **Data at Rest:** Mã hóa toàn bộ dữ liệu lưu trữ.
*   **Audit Logs:** Ghi log cho mọi thao tác ghi vào Long-Term Memory.
*   **GDPR/Compliance:** Có endpoint để xóa/trích xuất dữ liệu theo yêu cầu.
*   **Explicit Consent:** Vì chỉ persist khi hoàn tất, cần bước xác nhận cuối cùng: *"Xác nhận lưu thông tin?"*.

---

## 10. Pseudocode: Slot-Filling Loop

```python
def handle_user_message(session_id, user_text):
    # Lấy state từ Short-term memory
    slot_pool = redis.hget(session_id, "slot_pool") or {}
    
    # Stage 0: Fast patterns (Regex)
    slot_pool = fast_extract(user_text, slot_pool)
    
    # Stage 1: Light NER (nếu còn nhập nhằng)
    slot_pool = ner_extract_if_needed(user_text, slot_pool)
    
    # Kiểm tra độ hoàn thiện
    missing = [s for s in required_slots if not slot_pool[s].get("confirmed")]
    
    if missing:
        # Nếu thiếu: Tạo câu hỏi ngắn gọn
        question = build_minimal_question(missing, slot_pool)
        save_slot_pool(session_id, slot_pool)
        return agent_reply(question)
    else:
        # Nếu đủ: Xác nhận lần cuối
        confirm = ask_user_confirm(slot_pool)
        if confirm == True:
            # Ghi vào Long-term DB & Gọi API
            persist_episode(session_id, slot_pool, user_id)
            clear_session(session_id)
            return agent_reply("Đã lưu và gọi API thành công.")
        else:
            # User muốn sửa đổi -> Loop lại
            save_slot_pool(session_id, slot_pool)
            return agent_reply("Bạn muốn sửa trường nào?")
```

---

## 11. Ví dụ Prompt Clarifying Tối giản

**Mục tiêu:** Một câu hỏi, nhắm đúng field thiếu, tránh mở rộng hội thoại.

> "Mình thiếu số điện thoại và ngày để hoàn tất đặt vé. Bạn cho mình số điện thoại (ví dụ: 091234567) và ngày (YYYY-MM-DD) nhé?"

---

## 12. Telemetry & KPIs

*   **Avg Slots Filled / Session:** Hiệu quả thu thập thông tin.
*   **Time to Complete Episode:** Tốc độ hoàn thành tác vụ.
*   **Persisted vs Abandoned Rate:** Tỷ lệ chốt đơn thành công.
*   **Token Cost / Completed Episode:** Chi phí vận hành.
*   **Retrieval Precision:** Độ chính xác của thông tin gợi nhớ.

---

## 13. Đề xuất Tech Stack

*   **Runtime Agents:** Python + FastAPI.
*   **Short-Term Memory:** Redis (Hashes, TTL).
*   **Long-Term Memory:** Postgres (Metadata) + Vector DB (Pinecone/Weaviate/Chroma).
*   **Consensus (Optional):** Neo4j.
*   **Summarization:** Small LLM / Distilled Model (Background Job).

---

## 14. Quick Win Tips (Tối ưu thêm)

*   **No Raw Chat:** Tránh gửi toàn bộ hội thoại thô cho LLM, luôn gửi Slot Table + 1-sentence summary.
*   **Prompt Caching:** Cache output của các prompt có trạng thái slot giống hệt nhau (deduplication).
*   **Event-Driven:** Chạy tóm tắt Long-term khi trigger sự kiện `ready_to_persist` thay vì chạy định kỳ.
*   **Audit Endpoint:** Xây dựng endpoint để kỹ sư có thể kiểm tra dữ liệu *sắp* được persist (để debug an toàn).