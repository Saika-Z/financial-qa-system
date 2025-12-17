
---

# Financial QA System (RAG-based)

A professional financial question-answering system utilizing **BERT** for fine-tuned information extraction and **RAG (Retrieval-Augmented Generation)** for knowledge base querying. The system features a modern **Vue.js** frontend and a robust **Python** backend.

## 🚀 Quick Start

### 1. Prerequisites

* **Python 3.9+**
* **Docker & Docker Compose**
* **Node.js** (Optional, only for local frontend development)

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/financial-qa-system.git
cd financial-qa-system

```

### 3. Initialize Assets (Models & Data)

Since the model weights and processed knowledge bases are large, they are hosted on **Hugging Face**. Run the initialization script to sync these assets to your local machine:

```bash
pip install huggingface_hub
python init_project.py

```

*This script will automatically create the necessary directories and download the fine-tuned BERT model and the vectorized knowledge base.*

### 4. Deploy with Docker

Launch the entire stack (Frontend & Backend) using Docker Compose:

```bash
docker-compose up --build -d

```

* **Frontend:** Access via `http://localhost:8080`
* **Backend API:** Access via `http://localhost:8000`
* **API documentation** Access via `http://localhost:8000/docs`

---

## 🛠 Project Structure

```text
financial-qa-system/
├── frontend/             # Vue.js Project
│   ├── src/              # UI Components & Logic
│   └── Dockerfile        # Multi-stage build for Nginx
├── backend/              # Python FastAPI/Flask Backend
│   ├── data/kb/          # Vectorized Knowledge Base (Synced via script)
│   └── Dockerfile        # Python environment & API logic
├── models/               # Fine-tuned BERT Models (Synced via script)
├── init_project.py       # Asset synchronization script
└── docker-compose.yml    # Orchestration for the full stack

```

---

## 📊 Data & Models

* **Model:** Fine-tuned BERT on Kaggle financial datasets for NER and sentiment analysis.
* **Knowledge Base:** Processed financial news and terms stored in a vector database (FAISS/Milvus).
* **Data Source:** Scraped from public financial news outlets and Investopedia.

---

## 🔧 Configuration

The system uses environment variables for path management. You can modify these in the `docker-compose.yml`:

| Variable | Description | Default Path |
| --- | --- | --- |
| `MODEL_PATH` | Path to the BERT model directory | `/app/models/bert_finance` |
| `KB_PATH` | Path to the Knowledge Base directory | `/app/data/kb/finance_vector_db` |




---
## 📝 License

This project is licensed under the MIT License.

---
