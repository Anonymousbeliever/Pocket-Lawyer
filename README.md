# Pocket Lawyer Kenya

> An open-source, AI-powered legal companion for Kenya.

Pocket Lawyer Kenya is an AI-powered legal information system designed to make Kenyan legal knowledge easier to understand and access.

The system will provide answers grounded in authoritative Kenyan legal sources and will cite the legal documents used to generate each answer.

## Vision

Make reliable legal knowledge accessible to every Kenyan.

## Core Principles

- Accuracy over speculation
- Authoritative sources
- Traceable citations
- Privacy by design
- Open source
- Human lawyers remain essential
- AI assists with legal information; it does not replace advocates

## Project plan

Phases, current position, and progress are tracked in [`docs/project-plan.md`](docs/project-plan.md).

Legal source inventory: [`docs/legal-data/sources.md`](docs/legal-data/sources.md).

## Technology

### Backend
- Python
- FastAPI

### AI
- Retrieval-Augmented Generation (RAG)
- Embeddings
- Vector search
- Large Language Models

### Data
- Kenyan legislation
- Constitution
- Regulations
- Court judgments
- Other authoritative legal sources

### Mobile
- Flutter

## Project Structure

```text
pocket-lawyer/
├── backend/
├── data/
├── data-pipeline/
├── docs/
├── mobile/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt