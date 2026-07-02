![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Gemini](https://img.shields.io/badge/Google-Gemini_3.5_Flash-4285F4.svg)
![DeepL](https://img.shields.io/badge/DeepL-API-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

# Student Learning Journey Summarization System

**AI-Powered Educational Data Analytics Platform**

**Streamlit • Google Gemini 2.5 Flash • DeepL Translation • OCR • Docker • Production Ready**

**Demo Video:** https://youtu.be/pLLHletb9pE

**Sample Dataset:** https://github.com/user-attachments/files/29584548/Sample.Data.zip

**Sample Report:** https://drive.google.com/file/d/1g4QveYbhyFTo_bqDF5yw21BqoJC2DnnO/view?usp=sharing

---

## Overview

The **Student Learning Journey Summarization System** is an AI-powered educational analytics platform that transforms raw academic records into meaningful insights, multilingual reports, and professional AI-generated summaries.

Built with **Streamlit**, powered by **Google Gemini 2.5 Flash**, and integrated with the **DeepL Translation API**, the application processes both structured datasets and unstructured academic documents to generate comprehensive student learning reports.

The platform automates data cleaning, OCR processing, analytics, behavioral analysis, interactive visualization, AI-powered summarization, and multilingual translation, enabling educators and institutions to make informed, data-driven decisions.

Supported report languages include:

- English
- Bahasa Melayu
- Chinese

---

## Features

| Feature | Description |
|---|---|
| Structured Data Processing | Cleans and processes CSV/Excel academic datasets |
| Unstructured Document Processing | Extracts data from scanned documents using OCR |
| AI Executive Summary | Generates professional summaries using Google Gemini 2.5 Flash |
| DeepL Translation | Translates reports into multiple languages using DeepL API |
| Interactive Analytics | Visualizes student performance through dynamic charts and dashboards |
| Behavioral Analysis | Identifies learning patterns and behavioral trends |
| Student Information Extraction | Automatically extracts student profiles from raw documents |
| Secure Gemini Client | Handles AI requests with secure API integration |
| PII Protection | Protects sensitive student information before AI processing |
| Audit Logging | Records system activities for traceability |
| Docker Support | Fully containerized for consistent, reproducible deployment |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| AI Model | Google Gemini 2.5 Flash |
| Translation | DeepL API |
| Data Processing | Pandas |
| OCR | Pillow |
| Visualization | Plotly / Matplotlib |
| Containerization | Docker & Docker Compose |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (for containerized deployment)
- Google Gemini API key
- DeepL API key

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/student-learning-journey.git
cd student-learning-journey
```

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Then edit `.env` with your API keys (see [Environment Configuration](#environment-configuration)).

**5. Run the application**

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

---

## Environment Configuration

Create a `.env` file in the project root.

```env
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key

# DeepL Translation API
DEEPL_API_KEY=your_deepl_api_key
```

The application requires both API keys:

- **Google Gemini API** for AI-powered analysis and executive summary generation.
- **DeepL API** for multilingual report translation.

---

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker compose up --build
```

### Using Docker directly

```bash
docker build -t student-learning-journey .

docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_gemini_api_key \
  -e DEEPL_API_KEY=your_deepl_api_key \
  student-learning-journey
```

The application will be available at `http://localhost:8501`.

---

## Data Processing Pipeline

### Structured Data (CSV / Excel)

```
1.  Upload structured dataset
2.  Validate and inspect data
3.  Clean and preprocess records
4.  Detect and handle missing values
5.  Normalize and encode features
6.  Generate statistical summaries
7.  Perform behavioral analysis
8.  Render interactive visualizations
9.  Extract student profile information
10. Generate AI Executive Summary using Gemini
11. Translate the report using DeepL
12. Export the final report
```

### Unstructured Documents (Scanned / OCR)

```
1. Upload scanned academic document
2. Preprocess image for OCR
3. Extract raw text via OCR
4. Parse and structure extracted data
5. Perform behavioral and performance analysis
6. Generate AI Executive Summary
7. Translate into the selected language
8. Generate the final report
```

---

## Production Features

- Google Gemini 2.5 Flash integration
- DeepL Translation API integration
- Secure API key management via environment variables
- Docker and Docker Compose deployment
- OCR document processing
- AI-powered executive summaries
- Student information extraction
- Interactive analytics dashboards
- PII protection before AI processing
- Audit logging for system traceability
- Modular backend architecture
- Scalable production-ready design

---

## Future Improvements

- User authentication and authorization
- Cloud-native deployment (AWS, Azure, GCP)
- Historical student performance tracking
- Batch document processing
- REST API for third-party integration
- Database support (PostgreSQL / MySQL)
- Role-based access control
- Real-time analytics dashboard

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
