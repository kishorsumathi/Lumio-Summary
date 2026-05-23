from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from uuid import UUID

import psycopg2
import psycopg2.extras
import streamlit as st


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def _conn_params() -> dict[str, Any]:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "lumio"),
        "user": os.getenv("DB_USER", os.getenv("USER", "")),
        "password": os.getenv("DB_PASSWORD", ""),
    }


@st.cache_resource
def get_connection():
    return psycopg2.connect(**_conn_params())


def _cursor():
    conn = get_connection()
    try:
        conn.cursor().execute("SELECT 1")
    except Exception:
        conn = psycopg2.connect(**_conn_params())
        st.cache_resource.clear()
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PatientRow:
    clientid: UUID
    first_name: str
    last_name: str
    custom_patient_id: str
    date_of_birth: date | None
    gender: str | None

    @property
    def display_name(self) -> str:
        return f"{self.first_name} {self.last_name} ({self.custom_patient_id})"

    @property
    def age(self) -> str:
        if not self.date_of_birth:
            return ""
        today = date.today()
        years = today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
        return str(years)

    @property
    def age_gender(self) -> str:
        parts = [p for p in [self.age, self.gender] if p]
        return " / ".join(parts) if parts else ""


@dataclass
class SessionRow:
    session_id: UUID
    session_type: str
    session_number: int
    session_date: date
    modality: str
    status: str
    title: str | None
    clinician: str
    transcript: list[dict]
    transcript_tracks: dict | None
    start_time: datetime | None

    @property
    def modality_label(self) -> str:
        return {
            "in_person": "In-person",
            "video": "Online (Video)",
            "phone": "Online (Phone)",
            "async_message": "Async Message",
        }.get(self.modality, self.modality.replace("_", " ").title())

    @property
    def suggested_template(self) -> str:
        return {
            "intake": "Intake",
            "regular_session": "Regular Therapy",
            "check_in": "Regular Therapy",
            "emergency": "Emergency",
            "accommodations": "Section Change",
            "cogsash": "Follow Up Psychiatric Consultation",
        }.get(self.session_type, "Regular Therapy")

    @property
    def display_label(self) -> str:
        return (
            f"Session {self.session_number} — "
            f"{self.session_date.strftime('%d %b %Y')} — "
            f"{self.session_type.replace('_', ' ').title()}"
        )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def fetch_patients() -> list[PatientRow]:
    """All clients who have at least one session with a transcript."""
    cur = _cursor()
    cur.execute("""
        SELECT DISTINCT
            c.clientid,
            c.first_name,
            c.last_name,
            c.custom_patient_id,
            c.date_of_birth,
            c.gender
        FROM clients c
        JOIN sessions s ON c.clientid = s.client_id
        JOIN session_highlights sh ON sh.session_id = s.session_id
        WHERE sh.transcript IS NOT NULL
          AND jsonb_array_length(sh.transcript) > 0
        ORDER BY c.last_name, c.first_name
    """)
    rows = cur.fetchall()
    return [PatientRow(**r) for r in rows]


@st.cache_data(ttl=300)
def _has_column(table: str, column: str) -> bool:
    cur = _cursor()
    cur.execute("""
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s AND column_name = %s
    """, (table, column))
    return cur.fetchone() is not None


def fetch_sessions(client_id: UUID) -> list[SessionRow]:
    """All sessions with transcripts for a given patient, newest first."""
    cur = _cursor()
    has_tracks = _has_column("session_highlights", "transcript_tracks")
    tracks_col = "sh.transcript_tracks" if has_tracks else "NULL::jsonb AS transcript_tracks"

    cur.execute(f"""
        SELECT
            s.session_id,
            s.session_type,
            s.session_number,
            s.session_date,
            s.modality,
            s.status,
            s.title,
            s.start_time,
            p.firstname || ' ' || p.lastname AS clinician,
            sh.transcript,
            {tracks_col}
        FROM sessions s
        JOIN providers p ON s.providers_id = p.providers_id
        JOIN session_highlights sh ON sh.session_id = s.session_id
        WHERE s.client_id = %s
          AND sh.transcript IS NOT NULL
          AND jsonb_array_length(sh.transcript) > 0
        ORDER BY s.session_date DESC, s.session_number DESC
    """, (str(client_id),))
    rows = cur.fetchall()
    return [
        SessionRow(
            session_id=r["session_id"],
            session_type=r["session_type"],
            session_number=r["session_number"],
            session_date=r["session_date"],
            modality=r["modality"],
            status=r["status"],
            title=r["title"],
            start_time=r["start_time"],
            clinician=r["clinician"],
            transcript=r["transcript"] if isinstance(r["transcript"], list) else [],
            transcript_tracks=r["transcript_tracks"] if isinstance(r["transcript_tracks"], dict) else None,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Transcript assembly
# ---------------------------------------------------------------------------

def _get_chunks(session: SessionRow) -> list[dict]:
    """Extract transcript chunks from scribe_v2 track → sarvam track → raw transcript column."""
    if session.transcript_tracks:
        tracks = session.transcript_tracks.get("tracks", {})
        for track_name in ("scribe_v2", "sarvam"):
            rows = tracks.get(track_name, {}).get("rows")
            if rows:
                return rows
    return session.transcript


def assemble_transcript(session: SessionRow, use_translation: bool = True) -> str:
    """Convert transcript chunks → readable speaker-labelled text.

    Source priority: scribe_v2 track → sarvam track → transcript column.
    use_translation=True → use translation field, fall back to text if absent.
    use_translation=False → always use original text field.
    """
    chunks = _get_chunks(session)
    lines: list[str] = []
    for chunk in chunks:
        speaker = chunk.get("speaker") or "Unknown"
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        if use_translation:
            translation = (chunk.get("translation") or "").strip()
            lines.append(f"{speaker}: {translation if (translation and translation != text) else text}")
        else:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)
