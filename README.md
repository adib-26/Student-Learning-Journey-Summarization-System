#  Student Learning Journey Summarization System

AI-Powered Educational Data Analytics Platform
Dockerized • Gemini Integrated • Production Ready

Sample Data - [Sample Data.zip](https://github.com/user-attachments/files/25312898/Sample.Data.zip)

---

##  Overview

The **Student Learning Journey Summarization System** is a Streamlit-based educational analytics platform that processes both structured and unstructured academic data.

It transforms raw student records into:

*  Performance dashboards
*  Top 5 subject insights
*  Trend & predictive analytics
*  Behavioral trait extraction
*  AI-generated executive summaries (via Gemini API)

The system is fully **Dockerized** and designed for scalable deployment.

---

##  Key Features

| Feature                 | Description                              |
| ----------------------- | ---------------------------------------- |
| File Upload             | CSV, XLSX, PDF, PNG, JPG support         |
| OCR Processing          | Extracts text from scanned documents     |
| Data Cleaning           | Automated normalization & validation     |
| Analytics Engine        | Statistics, trend detection, predictions |
| Top 5 Analysis          | Identifies highest performing subjects   |
| Student Info Extraction | Detects student details automatically    |
| Behaviour Analysis      | Extracts behavior traits from reports    |
| Visualizations          | Dynamic charts and dashboards            |
| AI Executive Summary    | Professional summary using Gemini        |
| Docker Deployment       | Production-ready container setup         |

---

##  Project Structure

```
.
├── app.py
├── backend/
│   ├── analytics.py
│   ├── behaviour_extractor.py
│   ├── chart.py
│   ├── data_cleaning.py
│   ├── data_loader.py
│   ├── normalizer.py
│   ├── ocr_parser.py
│   ├── student_info_extractor.py
│   ├── summarizer.py
│   ├── text_info_extractor.py
│   ├── top5.py
│   └── ui_animations.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

##  Technology Stack

| Layer            | Technology        |
| ---------------- | ----------------- |
| Frontend         | Streamlit         |
| Backend          | Python            |
| Data Processing  | Pandas            |
| OCR              | PIL               |
| AI Model         | Google Gemini API |
| Containerization | Docker            |

---

##  Environment Variables

Create a `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Or pass it directly when running Docker.

---

##  Docker Setup

### 1️⃣ Build Image

```
docker build -t student-analytics-app .
```

### 2️⃣ Run Container

```
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_gemini_api_key_here \
  student-analytics-app
```

### 3️⃣ Open in Browser

```
http://localhost:8501
```

---

## 📂 Supported File Types

| Extension    | Purpose                    |
| ------------ | -------------------------- |
| .csv         | Structured grade datasets  |
| .xlsx / .xls | Academic spreadsheets      |
| .pdf         | Certificates & reports     |
| .png / .jpg  | Scanned academic documents |

---
## 🖼️ Application Preview

<img width="1319" height="646" alt="Screenshot 2026-05-02 at 3 53 19 PM" src="https://github.com/user-attachments/assets/a0b58878-0ebb-4ee2-94a5-553fa4e7795d" />
<img width="1310" height="482" alt="Screenshot 2026-05-02 at 3 54 04 PM" src="https://github.com/user-attachments/assets/a63ffbe2-321b-4e62-9284-09b348a05d01" />
<img width="1284" height="521" alt="Screenshot 2026-05-02 at 3 54 43 PM" src="https://github.com/user-attachments/assets/2e414fef-79cf-4878-a7d4-1603eeda5c4a" />
<img width="1306" height="588" alt="Screenshot 2026-05-02 at 3 54 57 PM" src="https://github.com/user-attachments/assets/32f52612-3d3f-4317-8817-2e8f6832978d" />

---

## 🔄 Data Processing Flow

### Structured Data

1. Upload dataset
2. Normalize score columns
3. Clean dataframe
4. Compute statistics
5. Detect trends
6. Generate predictive insights
7. Render visualizations
8. Generate AI summary

### Unstructured Data

1. OCR extraction
2. Student name detection
3. Certificate parsing via Gemini
4. Skill extraction
5. AI executive summary

---

## 📊 Dashboard Metrics

* Average Score
* Total Records
* Highest Score
* Lowest Score
* Top 5 Subjects
* Performance Charts
* Professional Executive Summary

---

## 🎯 Use Cases

* Schools & Universities
* Academic Advisors
* Educational Consultants
* EdTech Platforms
* Student Portfolio Analytics

---

## 🔒 Production Notes

* API keys stored via environment variables
* No hardcoded credentials
* Fully containerized
* Ready for cloud deployment (AWS / GCP / Azure)

---

## 👨‍💻 Author

Mahbub
Educational Data Analytics Platform

---

## 📄 License

This project is intended for educational and analytics use.


