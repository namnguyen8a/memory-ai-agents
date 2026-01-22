# Mem0 Memory Architecture with LangGraph Multi-Agent System

Đây là một implementation demo của kiến trúc memory Mem0 được tích hợp với hệ thống multi-agent sử dụng LangGraph.

## 📋 Mục lục

1. [Tổng quan](#tổng-quan)
2. [Kiến trúc](#kiến-trúc)
3. [Pipeline Chi tiết](#pipeline-chi-tiết)
4. [Cài đặt](#cài-đặt)
5. [Sử dụng](#sử-dụng)
6. [Production Notes](#production-notes)
7. [Cấu trúc dự án](#cấu-trúc-dự-án)

---

## 🎯 Tổng quan

### Mem0 Memory Architecture

Mem0 là một kiến trúc memory được thiết kế để:
- **Nén hội thoại thành facts**: Chỉ lưu những sự thật quan trọng, không lưu toàn bộ hội thoại
- **Vector Search**: Sử dụng vector database để tìm kiếm memories liên quan theo ngữ nghĩa
- **Function Calling**: Sử dụng LLM để quyết định ADD/UPDATE/DELETE/NOOP memories
- **Hiệu quả cao**: Target latency < 0.70s, tiết kiệm ~90% chi phí so với full context RAG

### Multi-Agent System

Hệ thống multi-agent bao gồm:
- **Supervisor Agent**: Route tasks đến các agent chuyên biệt
- **Sales Agent**: Xử lý inquiries về sản phẩm
- **Support Agent**: Xử lý technical support
- **General Agent**: Xử lý câu hỏi chung

### Demo Features

- ✅ File-based vector store (mock cho demo)
- ✅ File-based conversation storage (mock cho demo)
- ✅ Mock embeddings (random vectors)
- ✅ LLM-based fact extraction
- ✅ LLM-based memory update decisions
- ✅ Multi-agent routing

---

## 🏗️ Kiến trúc

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              SUPERVISOR AGENT (LangGraph)                    │
│              Routes to appropriate agent                      │
└────────┬───────────────┬───────────────┬────────────────────┘
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ SALES AGENT │  │SUPPORT AGENT│  │GENERAL AGENT│
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              MEM0 MEMORY MANAGER                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Stage 1: RETRIEVAL & GENERATION (Hot Path)          │  │
│  │  - Vector Search (Top-K memories)                    │  │
│  │  - Generate response with memories                   │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Stage 2: EXTRACTION & UPDATE (Cold Path)            │  │
│  │  - Extract facts from conversation                    │  │
│  │  - Decide operation (ADD/UPDATE/DELETE/NOOP)         │  │
│  │  - Update vector store                                │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Stage 3: ASYNC SUMMARIZATION (Background)           │  │
│  │  - Update global summary periodically                 │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
               ▼                      ▼
    ┌──────────────────┐    ┌──────────────────┐
    │  Vector Store    │    │  Storage         │
    │  (Memories)      │    │  (Context)       │
    │                  │    │                  │
    │  ⚠️ Demo: File   │    │  ⚠️ Demo: File   │
    │  Production:     │    │  Production:     │
    │  Qdrant/Pinecone │    │  Redis/Postgres  │
    └──────────────────┘    └──────────────────┘
```

### Data Flow

```
USER QUERY
    │
    ├──► Supervisor Node ──► Route to Agent
    │                           │
    │                           ▼
    │                    Agent Node
    │                           │
    │                           ├──► Retrieve Memories (Vector Search)
    │                           │           │
    │                           │           ▼
    │                           │    Get Top-K Memories
    │                           │           │
    │                           │           ▼
    │                           ├──► Generate Response (with memories)
    │                           │           │
    │                           │           ▼
    │                           │    Return Response
    │                           │           │
    │                           ▼           │
    ├──► Memory Update Node ◄──┘           │
    │         │                             │
    │         ├──► Extract Facts            │
    │         │         │                   │
    │         │         ▼                   │
    │         │    Get Facts List           │
    │         │         │                   │
    │         ├──► For each fact:           │
    │         │         │                   │
    │         │         ├──► Search Similar Memories
    │         │         │         │
    │         │         │         ▼
    │         │         ├──► Decide Operation (LLM)
    │         │         │         │
    │         │         │         ▼
    │         │         └──► Execute (ADD/UPDATE/DELETE/NOOP)
    │         │
    │         └──► Update Conversation Context
    │
    ▼
RESPONSE TO USER
```

---

## 🔄 Pipeline Chi tiết

### Stage 1: Retrieval & Generation (Hot Path)

**Mục tiêu**: Trả lời User nhanh nhất có thể (Target: < 1.5s)

1. **Vector Search**:
   - Embed query text
   - Search vector DB với filter `user_id`
   - Lấy Top-K (k=10)

2. **Generation**:
   - Đưa memories vào System Prompt
   - LLM generate response sử dụng memories
   - Return response cho User

**Latency Target**: ~0.70s total

### Stage 2: Extraction & Update (Cold Path)

**Mục tiêu**: Ghi nhớ thông tin mới, loại bỏ tin cũ/sai

1. **Extraction (LLM Call 1)**:
   - Trích xuất facts từ conversation
   - Input: (Q, A) + Global Summary + Recent Messages
   - Output: List of facts `Ω = [f1, f2, ...]`

2. **Update Loop** (cho mỗi fact):
   - **Step 3a**: Search tương đồng trong Vector DB
   - **Step 3b**: Function Calling (LLM Call 2)
     - Quyết định: ADD / UPDATE / DELETE / NOOP
   - **Step 3c**: Execute operation

**Latency**: Có thể chạy async/background để không block response

### Stage 3: Async Summarization (Background)

**Trigger**: Chạy sau mỗi N lượt hội thoại

**Action**: Cập nhật Global Summary để phục vụ cho Stage 2 lần sau

---

## 📦 Cài đặt

### 1. Clone và cài đặt dependencies

```bash
cd langgraph-memory/multi-agents-memory/mem-zero/implement

# Tạo virtual environment (khuyến nghị)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Cấu hình API Key

Tạo file `.env` trong thư mục `implement/`:

```env
# Chọn provider: openai (default) hoặc ollama
LLM_PROVIDER=ollama

# Nếu dùng Ollama (local)
# - Tool-calling model: functiongemma:270m
# - Normal model: hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest
OLLAMA_MODEL=functiongemma:270m

# Nếu dùng OpenAI
OPENAI_API_KEY=your_openai_api_key_here
```

Hoặc set environment variable:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Chạy demo

```bash
python demo.py
```

**Lưu ý với Ollama**
- Cài Ollama và pull model trước:
  - `ollama pull functiongemma:270m`
  - `ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest`
- Đặt `LLM_PROVIDER=ollama` và chọn `OLLAMA_MODEL` mong muốn.

---

## 🚀 Sử dụng

### Basic Usage

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.memory.memory_manager import MemoryManager
from src.memory.vector_store import FileVectorStore
from src.memory.storage import FileStorage
from src.agents.graph import create_multi_agent_graph
import uuid

# Load environment
load_dotenv()

# Initialize components
vector_store = FileVectorStore(storage_path="./data/vector_store")
storage = FileStorage(storage_path="./data/storage")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory_manager = MemoryManager(vector_store, storage, llm)

# Create graph
graph = create_multi_agent_graph(memory_manager, llm)

# Create initial state
user_id = "user_123"
session_id = f"session_{uuid.uuid4().hex[:8]}"

initial_state = {
    "messages": [HumanMessage(content="I need help with a product")],
    "next_agent": "general",
    "user_id": user_id,
    "session_id": session_id,
    "agent_id": "general",
    "memory_context": {}
}

# Run graph
result = graph.invoke(initial_state)

# Get response
response = result["messages"][-1].content
print(f"Response: {response}")
```

### Test Scenarios

Demo script (`demo.py`) test các scenarios:
1. Sales agent routing
2. Support agent routing
3. General agent routing
4. Memory storage (user preferences)
5. Memory retrieval
6. Memory update (contradictions)

### Interactive Testing (Turn 1..N)

Bạn có thể test memory liên tục theo turn (vd 1..10) bằng chế độ interactive:

```bash
python demo.py
```

- Gõ message và nhấn Enter để chạy 1 turn
- Gõ `q` hoặc `quit` để thoát
- Log sẽ hiển thị:
  - **[TIME]** cho từng bước (search/generate/extract/update…)
  - **[ROUTE]** supervisor route sang agent nào
  - **[MEMORY_RETRIEVED]** số lượng memory id được retrieve
  - **[CONTEXT]** số lượng recent_messages (rolling window tối đa 10)

---

## ⚠️ Production Notes

### Components cần thay thế cho Production

#### 1. Vector Store

**Demo**: File-based mock với random embeddings

**Production**: 
- Qdrant (recommended)
- Pinecone (cloud-based)
- ChromaDB (open-source)
- Weaviate (self-hosted)
- pgvector (PostgreSQL extension)

**Code location**: `src/memory/vector_store.py`

#### 2. Embedding Model

**Demo**: Random vectors (mock)

**Production**:
- OpenAI: `text-embedding-3-small` (1536 dim) hoặc `text-embedding-3-large` (3072 dim)
- Sentence Transformers: `sentence-transformers/all-MiniLM-L6-v2` (384 dim)
- Google: `textembedding-gecko@003`

**Code location**: `src/memory/memory_manager.py::_generate_embedding()`

#### 3. Conversation Storage

**Demo**: File-based JSON storage

**Production**:
- Redis (cho session-level data với TTL)
- PostgreSQL (cho persistent conversation data)
- MongoDB (cho flexible schema)

**Code location**: `src/memory/storage.py`

#### 4. Async Processing

**Demo**: Synchronous execution (memory update blocks response)

**Production**:
- Run memory update in background (Celery, RQ, Background Tasks)
- Use queue (Redis Queue) để serialize updates per user_id
- Hot Path (response) và Cold Path (update) nên chạy song song

**Code location**: `src/agents/graph.py::memory_update_node()`

#### 5. Summarization

**Demo**: Not implemented (placeholder)

**Production**:
- Background job (Celery, Cron, etc.)
- Run sau mỗi N turns (e.g., 5-10 turns)
- Update global summary để tối ưu extraction

**Code location**: `src/memory/memory_manager.py::summarize_conversation()`

### Production Checklist

- [ ] Replace file-based vector store với Qdrant/Pinecone
- [ ] Replace mock embeddings với OpenAI/Sentence Transformers
- [ ] Replace file storage với Redis/PostgreSQL
- [ ] Implement async memory update (background tasks)
- [ ] Implement async summarization (background jobs)
- [ ] Add error handling và retry logic
- [ ] Add logging và monitoring
- [ ] Add rate limiting
- [ ] Add authentication/authorization
- [ ] Add metrics và observability
- [ ] Add tests (unit, integration, e2e)
- [ ] Add deployment configs (Docker, Kubernetes, etc.)

---

## 📁 Cấu trúc dự án

```
implement/
├── src/
│   ├── __init__.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── models.py              # Data models (MemoryItem, ConversationContext, MemoryOperation)
│   │   ├── vector_store.py        # File-based vector store (⚠️ Demo)
│   │   ├── storage.py             # File-based storage (⚠️ Demo)
│   │   └── memory_manager.py      # Core mem0 pipeline implementation
│   └── agents/
│       ├── __init__.py
│       ├── state.py               # LangGraph state schema
│       └── graph.py               # Multi-agent graph definition
├── demo.py                        # Demo/test script
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
└── README.md                      # This file
```

---

## 📚 Tài liệu tham khảo

- [Mem0 Architecture Note](./architecture_note.md)
- [LangGraph Multi-Agent Documentation](https://langchain-ai.github.io/langgraph/how-tos/multi_agent/)
- [LangGraph Memory Documentation](https://langchain-ai.github.io/langgraph/how-tos/memory/)

---

## 🔧 Troubleshooting

### Issue: OPENAI_API_KEY not found

**Solution**: Set `OPENAI_API_KEY` in `.env` file or environment variable

### Issue: Import errors

**Solution**: Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Data not persisting

**Solution**: Check that `./data/` directory is writable. Data is stored in:
- `./data/vector_store/` - Vector database files
- `./data/storage/` - Conversation context files

---

## 📝 Notes

- Đây là một **demo implementation** để test feasibility
- Một số components là **mock** (file-based storage, random embeddings)
- Cho production, cần thay thế với real implementations (see Production Notes)
- Code có **comments chi tiết** để chỉ rõ production requirements

---

## 🎯 Next Steps

1. Test với real embeddings (OpenAI, Sentence Transformers)
2. Test với real vector DB (Qdrant, Pinecone)
3. Implement async memory update
4. Implement async summarization
5. Add comprehensive tests
6. Deploy to production environment

---

**Version**: 0.1.0  
**Last Updated**: 2025-01-30
