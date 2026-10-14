# Automation Python Backend

[![tests](https://github.com/spraicheux/automation-python-backend/actions/workflows/tests.yml/badge.svg)](https://github.com/spraicheux/automation-python-backend/actions/workflows/tests.yml)

Async FastAPI backend deployed on Azure App Service.

This service is designed to process Email and WhatsApp offers via Make.com without timeout by using a fully asynchronous, non-blocking architecture.

---

## Overview

The backend provides:

- Fast ingestion endpoint
- Background async processing
- OpenAI-powered document parsing
- Structured JSON output
- Redis-based job tracking
- Non-blocking request handling

The system immediately accepts jobs and processes them asynchronously to prevent HTTP timeouts.

---

## Architecture Flow

Make.com (Client)
→ POST /ingest
→ Job stored in Redis
→ Background worker processes document
→ OpenAI parsing
→ Result stored in Redis
→ GET /result/{job_id}

### Job Lifecycle

accepted → processing → done

This ensures:
- No blocking synchronous execution
- Scalable async processing
- Safe status tracking

---

## API Endpoints

### POST /ingest

Accepts input file or structured payload and immediately returns a job ID.

Example Response:

{
  "job_id": "uuid",
  "status": "accepted"
}

---

### GET /result/{job_id}

Returns job status and final structured output.

Example Response:

{
  "job_id": "uuid",
  "status": "done",
  "data": { ... }
}

---

## Interactive API Documentation

FastAPI automatically provides interactive documentation:

Swagger UI:
https://whatsapp-automation-backend-app-cqd2fteqh6hvhped.francecentral-01.azurewebsites.net/docs#/System/health_health_get


---

## Technology Stack

- Python 3.11
- FastAPI
- Azure App Service
- Azure Redis Cache
- OpenAI API
- Make.com integration

---

## Environment Variables

All sensitive configuration is managed via Azure App Service Configuration.

Required variables:

- OPENAI_API_KEY
- REDIS_HOST
- REDIS_PORT
- REDIS_PASSWORD
- AZURE_ENV

No secrets are hardcoded in the repository.

---

## Deployment

Deployment is handled via CI/CD.

On push to main:
- GitHub Actions triggers build
- Azure App Service auto-deploys
- Environment variables are injected from Azure

---

## Async Processing Design

This backend avoids timeout issues by:

- Returning immediately on ingestion
- Processing heavy tasks in background workers
- Using Redis for state persistence
- Never performing blocking synchronous calls in API routes

This ensures full compatibility with Make.com timeout limits.

---

## Project Structure

api/                → Route definitions  
core/               → OpenAI client logic  
schemas/            → Request/response schemas  
workers/            → Background processing  
main.py             → FastAPI entrypoint  
requirements.txt    → Dependencies  
runtime.txt         → Python runtime version  
azure-startup.sh    → Azure startup configuration  

---

## Production Notes

- No hardcoded secrets
- Fully async implementation
- Redis-backed job lifecycle tracking
- Clean, production-ready structure
- Designed for scalability and reliability

