# VHS Analyzer — Veterinary Heart Score Interpretation Tool

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey)
![Vue.js](https://img.shields.io/badge/Vue.js-3-42b883)
![Docker](https://img.shields.io/badge/Docker-ready-2496ed)

A full-stack application that analyzes Vertebral Heart Score (VHS) measurements for veterinary cardiology. Combines a PyTorch CNN for radiograph analysis with an LLM for clinical interpretation.

## Features

- **Radiograph upload** — automatic extraction of L, S, T measurements via EfficientNet-B7
- **Manual entry** — enter measurements directly without an image
- **AI interpretation** — species and breed-specific clinical assessment powered by Llama 3.3 (Groq)
- **Editable results** — modify or complete the AI-generated interpretation before export
- **PDF export** — save the full analysis report as a PDF

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Vue.js 3, Vite, Nginx |
| Backend | Flask 3, Gunicorn |
| Deep Learning | PyTorch, EfficientNet-B7 |
| LLM | Llama 3.3-70b via Groq + LangChain |
| Containerization | Docker, Docker Compose |

## How It Works

### Image Analysis
A chest radiograph is processed through EfficientNet-B7, which detects 6 key points representing:
- **L** — long axis from carina to cardiac apex
- **S** — short axis perpendicular to L at the widest part of the heart
- **T** — reference vertebral length

### VHS Calculation
```
VHS = 6 × ((L + S) / T)
```
Normal ranges:
- Dogs: 9.7 ± 0.5 vertebrae (8.7–10.7)
- Cats: 7.5 ± 0.3 vertebrae (7.0–8.1)

### Clinical Interpretation
Llama 3.3-70b analyzes the VHS score considering animal type, breed, age, weight, and sex. It returns a structured report with normal range assessment, severity classification, possible conditions, and recommendations.

## Quick Start (Docker)

The fastest way to run the app — no code required, just Docker and a `.env` file.

```bash
# 1. Get the compose file
curl -O https://raw.githubusercontent.com/HermesNdjeng/project_ai_health/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/HermesNdjeng/project_ai_health/main/.env.example

# 2. Configure your API keys
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY and MODEL_URL

# 3. Run
docker compose up
```

The app is available at [http://localhost](http://localhost).

## Local Development

**Prerequisites:** Python 3.11+, Poetry, Node.js 20+

```bash
# Clone
git clone https://github.com/HermesNdjeng/project_ai_health.git
cd project_ai_health

# Backend
poetry install
cp .env.example .env  # fill in your keys
python app.py

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key from [console.groq.com](https://console.groq.com) |
| `MODEL_URL` | Direct download URL for the `.pt` model file |

## Running Tests

```bash
pytest tests/ -v
```

## Screenshots

<!-- Add screenshots of the app here -->

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Developed as part of the AI for Health course at Aivancity.*
