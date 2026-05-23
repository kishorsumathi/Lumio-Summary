# Lumio Clinical Summary Generator

A Streamlit app that connects to the Lumio PostgreSQL database, retrieves patient sessions and transcripts, and generates structured clinical notes using Claude (Sonnet / Opus) via LangChain.

## Features

- **DB-driven workflow** — select a patient and session directly from the Lumio database; no file uploads needed
- **Transcript preview** — toggle between translated and original text before generating
- **Template-matched output** — each session type maps to a Markdown template; the LLM fills the exact fields and structure
- **Smart transcript extraction** — prefers ElevenLabs `scribe_v2` track from `transcript_tracks`, falls back to `sarvam`, then the raw `transcript` column
- **Conditional prompt assembly** — dynamic sections 3 & 4 (controlled vocabularies + screening checklists) are injected for all session types except Room Change and Section Change
- **DOCX download** — generated summary exported as a formatted Word document

## Session Types

| Note Type | Template | Dynamic §3 & §4 |
|---|---|---|
| Intake | `format/Intake.md` | Yes |
| Regular Therapy | `format/Regular_Therapy.md` | Yes |
| Couples Therapy | `format/Couples_Therapy.md` | Yes |
| Emergency | `format/Emergency.md` | Yes |
| Follow Up Psychiatric Consultation | `format/Follow_Up_Psychiatric_Consultation.md` | Yes |
| Room Change | `format/Room_Change.md` | No |
| Section Change | `format/Section_Change.md` | No |

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager
- Python 3.12+
- PostgreSQL `lumio` database accessible locally

### Install dependencies

```bash
uv sync
```

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=sk-ant-...

DB_HOST=localhost
DB_PORT=5432
DB_NAME=lumio
DB_USER=your_pg_user
DB_PASSWORD=
```

### Run

```bash
uv run streamlit run app.py
```

## Project Structure

```
lumio-summary/
├── app.py                                      # Streamlit UI + LLM chain
├── db.py                                       # DB connection, data classes, queries
├── format/                                     # Markdown templates per session type
│   ├── Intake.md
│   ├── Regular_Therapy.md
│   ├── Couples_Therapy.md
│   ├── Emergency.md
│   ├── Follow_Up_Psychiatric_Consultation.md
│   ├── Room_Change.md
│   └── Section_Change.md
├── lumio_clinical_summary_system_prompt.txt    # Base clinical guardrails system prompt
├── lumio_dynamic_sections_3_4_prompt.txt       # Controlled vocabularies + screening checklists
├── pyproject.toml
└── uv.lock
```

## How It Works

1. **Patient & session selection** — fetches all clients with non-empty transcripts; selecting a patient loads their sessions
2. **Transcript assembly** — chunks are extracted from `transcript_tracks.tracks.scribe_v2.rows` (preferred) → `sarvam.rows` → `session_highlights.transcript`; each chunk uses the `translation` field, falling back to the original `text`
3. **Prompt assembly** — system prompt = base guardrails + (dynamic §3/§4 if applicable) + output formatting rules; user prompt = session type + template + transcript + static metadata
4. **Generation** — `ChatAnthropic.with_structured_output(ClinicalSummaryOutput)` returns a single `rendered_summary` string matching the template structure
5. **Export** — Markdown is parsed and written to a `.docx` via `python-docx`

## Models

Available in the sidebar:

| Label | Model ID |
|---|---|
| Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| Claude Opus 4.6 | `claude-opus-4-6` |
