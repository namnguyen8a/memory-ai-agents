# ARCHITECTURE & ENGINEERING: LLM AGENT MEMORY SYSTEMS
> **Tài liệu tổng hợp**: Các kỹ thuật, kiến trúc và chiến lược triển khai hệ thống bộ nhớ cho tác nhân AI (AI Agents) trong môi trường production.

---

## 1. CÁC HÌNH THÁI BỘ NHỚ CỐT LÕI (CORE PARADIGMS)

Hệ thống bộ nhớ của Agent được chia thành hai hình thái chính:

### A. Bộ nhớ Văn cảnh (Contextual/Textual Memory)
Thông tin được lưu trữ dưới dạng ngôn ngữ tự nhiên hoặc dữ liệu cấu trúc và đưa vào LLM qua prompt.
*   **Complete Interactions (Tương tác toàn bộ):** Lưu toàn bộ lịch sử. *Ưu điểm:* Toàn diện. *Nhược điểm:* Giới hạn context window, chi phí cao.
*   **Recent Interactions (Tương tác gần đây):** Sử dụng cơ chế cửa sổ trượt (sliding window) hoặc buffer theo "Nguyên tắc cục bộ".
*   **Retrieved Interactions (Tương tác được truy xuất):** Dùng Vector Database (RAG) để tìm kiếm các đoạn ký ức liên quan nhất theo ngữ nghĩa.
*   **Structured Contextual Memory:** Tổ chức dạng SQL, Knowledge Graph, hoặc Triples để suy luận logic phức tạp.

### B. Bộ nhớ Tham số (Parametric Memory)
Thông tin được "ngấm" vào trọng số (weights) của mô hình.
*   **Supervised Fine-Tuning (SFT):** Huấn luyện lại mô hình với dữ liệu chuyên biệt.
*   **Knowledge Editing:** Chỉnh sửa cục bộ các sự thật (facts) bằng kỹ thuật như ROME hoặc MEMIT mà không làm hỏng tri thức nền.
*   **KV Caching & Prompt Caching:** Lưu trạng thái Key-Value của token trong GPU để tăng tốc độ xử lý.

---

## 2. KIẾN TRÚC BỘ NHỚ SẢN PHẨM (PRODUCTION ARCHITECTURE)

Trong môi trường thực tế, bộ nhớ được tổ chức theo mô hình **Three-Tier (3 Lớp)**:

### Tier 1: Working Memory (Bộ nhớ làm việc)
*   **Phạm vi:** Nằm trong Context Window của LLM (4K-128K tokens).
*   **Đặc điểm:** Độ trễ cực thấp (<1ms), xử lý suy luận tức thời.
*   **Rủi ro:** Chi phí cao nhất, dễ bị "tràn bộ nhớ".

### Tier 2: Short-Term Memory (Bộ nhớ phiên/ngắn hạn)
*   **Phạm vi:** Cấp độ Session (Phiên làm việc).
*   **Công nghệ:** Redis, In-memory storage, LangGraph checkpointers.
*   **Chức năng:** Duy trì mạch hội thoại, state của người dùng hiện tại (độ trễ 10-50ms).

### Tier 3: Long-Term Memory (Bộ nhớ dài hạn)
*   **Phạm vi:** Liên phiên (Cross-session).
*   **Công nghệ:** Vector Database (Pinecone, Weaviate).
*   **Cấu trúc H-MEM (Hierarchical Memory):** Tổ chức theo 4 lớp trừu tượng để tối ưu truy xuất:
    1.  **Domain:** Miền (VD: CSKH).
    2.  **Category:** Danh mục (VD: Khiếu nại billing).
    3.  **Trace:** Dấu vết/Index tóm tắt.
    4.  **Episode:** Chi tiết hội thoại gốc.

---

## 3. CÁC KIẾN TRÚC ĐẶC BIỆT (SPECIALIZED ARCHITECTURES)

*   **MemGPT:** Mô phỏng hệ điều hành. Phân chia **Working Context** (RAM) và **External Storage** (Disk). Agent tự quản lý việc đẩy dữ liệu ra/vào hàng đợi.
*   **G-Memory (Cho Multi-Agent):** Quản lý qua 3 loại đồ thị:
    *   *Insight Graph:* Kinh nghiệm trừu tượng.
    *   *Query Graph:* Thông tin nhiệm vụ.
    *   *Interaction Graph:* Nhật ký giao tiếp chi tiết.
*   **Mentioned Slot Pool (MSP):** Chuyên dụng cho theo dõi trạng thái hội thoại (DST), ghi lại các giá trị (slot values) để tránh nhiễu thông tin cũ.

---

## 4. QUY TRÌNH TRÍCH XUẤT THÔNG TIN (EXTRACTION PIPELINE)

Để tối ưu chi phí và tốc độ, không sử dụng LLM cho mọi thứ. Quy trình gồm 3 giai đoạn:

1.  **Stage 1: Fast Pattern Matching (Regex/Rules)**
    *   Xử lý: Tên, Email, SĐT, cụm từ cố định ("I am", "I live in").
    *   Hiệu suất: <10ms, độ chính xác ~95%, chi phí $0.
2.  **Stage 2: Entity Recognition with Context (NER)**
    *   Xử lý: Giải quyết nhập nhằng ngữ nghĩa (VD: "ở đó" là Paris hay London?).
    *   Công nghệ: BERT/BiLSTM hoặc LLM nhỏ.
3.  **Stage 3: Slot Filling (Điền tham số)**
    *   Xử lý: Tổng hợp thông tin qua nhiều lượt hội thoại để gọi API (VD: Đặt vé -> Cần: Điểm đến, Ngày, Số người).

---

## 5. QUẢN LÝ VÀ TỐI ƯU HÓA (OPERATIONS & OPTIMIZATION)

### Chiến lược Truy xuất (Retrieval Strategy)
Công thức tính điểm liên quan (Relevance Score) cho ký ức:
```python
Score = (Semantic_Similarity * 0.4) +  # Độ tương đồng vector
        (Importance_Score * 0.3) +     # Mức độ quan trọng (đánh dấu)
        (Recency_Factor * 0.2) +       # Tính mới (Time decay)
        (Domain_Relevance * 0.1)       # Phù hợp ngữ cảnh
```

### Quản lý Ngữ cảnh (Context Management)
*   **Dynamic Budget Allocation:** Phân bổ token động (VD: 25% cho lịch sử, 15% cho tài liệu RAG, 10% dự phòng).
*   **Hierarchical Summarization:** Nén dữ liệu theo tầng:
    *   *Level 0:* Giữ nguyên 10 lượt gần nhất.
    *   *Level 1:* Tóm tắt trao đổi cũ.
    *   *Level 2:* Digest các chủ đề chính.
    *   *Level 3:* Profile người dùng dài hạn.
    *   *Kết quả:* Giảm tới 75% lượng token tiêu thụ.

### Các thao tác cốt lõi (Operations)
*   **Encoding:** Consolidation (Củng cố ngắn hạn -> dài hạn) & Indexing.
*   **Evolving:** Updating (Cập nhật) & Forgetting (Xóa tin cũ/sai).
*   **Adapting:** Retrieval & Condensation (Nén).

---

## 6. BỘ NHỚ TRONG HỆ THỐNG ĐA TÁC NHÂN (MULTI-AGENT MEMORY)

Hệ thống Multi-Agent cần các loại bộ nhớ chia sẻ để phối hợp:
*   **Consensus Memory:** Lưu các quy trình/protocol đã được cả team xác thực là hiệu quả.
*   **Persona Libraries:** Thư viện vai trò, tránh nhầm lẫn nhiệm vụ.
*   **Dependency Resolution:** Bộ nhớ trạng thái dependency (Task C chỉ chạy khi Task A & B xong).

---

## 7. ẨN DỤ TỔNG KẾT (THE LAWYER ANALOGY)

Hãy tưởng tượng hệ thống bộ nhớ Agent như bàn làm việc của một **Luật sư**:

*   **Bộ nhớ ngữ cảnh (Contextual):** Các hồ sơ đang mở trên bàn (Working) hoặc trong tủ hồ sơ ngay sau lưng (Short/Long-term RAG).
*   **Bộ nhớ tham số (Parametric):** Kiến thức luật đã học thuộc lòng trong đầu luật sư.
*   **Kiến trúc phân tầng (H-MEM):** Hệ thống nhãn dán hồ sơ: Hình sự -> Lừa đảo -> Vụ án Nguyễn Văn A.
*   **Cơ chế quên/Nén:** Luật sư xé bỏ giấy nháp không quan trọng, chỉ giữ lại biên bản cuối cùng.

---