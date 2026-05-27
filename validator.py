"""LLM-based validator agent for clinical summaries.

Runs after the main generation step as a second-pass gate. The validator
reviews the generated note against the v3.0 ruleset and returns a revised
summary plus the list of corrections it made.

Why LLM-only (no regex layer):
  - 7 session types, unlimited transcripts → new phrasings of every violation
    arise constantly. A regex matcher only catches phrases we've already named.
  - The validator needs to make clinical judgments (is this content tangential?
    is this formulation leaking from §1?) that pattern-matching cannot make.

Why the validator uses the SAME (or better) model as the generator:
  - The validator is the final gate. If it is less capable than the generator,
    it cannot reliably catch what the generator slipped through.
  - Cost roughly doubles per note — acceptable for a clinical workflow where
    accuracy is the point. Latency is the real trade-off; if it matters, the
    caller can override the model name.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class _Correction(BaseModel):
    """One specific violation the validator caught and corrected."""

    rule: str = Field(
        description=(
            "Short stable identifier for the rule that was violated. Use one of: "
            "empty_field_disclaimer, forbidden_quote, visual_cue_affect, "
            "side_effects_misplacement, target_symptom_tangential, "
            "formulation_leak, section_renumbering, medication_name_missing, "
            "repetition, risk_level_invented, speech_misuse, "
            "response_label_invented, temporal_precision, other."
        )
    )
    location: str = Field(
        description="The field or section affected, e.g. '§3 Sleep' or '§2 Side Effects'."
    )
    detail: str = Field(
        description=(
            "One-sentence human-readable description of the violation and what was done about it "
            "(removed, moved, rephrased)."
        )
    )


class _ValidatorOutput(BaseModel):
    revised_summary: str = Field(
        description=(
            "The full clinical summary with every violation corrected. Preserve everything "
            "that does not violate a rule — clinician voice, structure, and vocabulary unchanged. "
            "Return the complete note from the first heading to the last field."
        )
    )
    corrections: list[_Correction] = Field(
        description="One entry per violation you found. Empty list if the note is already clean.",
        default_factory=list,
    )


@dataclass
class ValidationIssue:
    """Public surface — matches what the UI consumes."""

    rule: str
    location: str
    detail: str


@dataclass
class ValidationResult:
    revised: str
    issues: list[ValidationIssue]

    @property
    def count(self) -> int:
        return len(self.issues)


# ---------------------------------------------------------------------------
# Validator prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the post-generation validator for the Lumio clinical-summary pipeline.
A clinician-facing note has already been generated from a session transcript.
Your job is to find violations of the rules below and return a corrected note.

You are not re-writing the note. You preserve everything that is rule-compliant —
clinician voice, structure, vocabulary, paragraph order — and only revise the
parts that violate a rule.

<rules>

R1. EMPTY-FIELD DISCLAIMERS — ANY WORDING.
   If a field's content announces that the field has no real content — in any
   wording, any tense — the entire field must be REMOVED (not just the
   disclaimer). Examples of violations:
     - "Not assessed" / "Not discussed" / "Not specified" / "Not reported" / "N/A"
     - "No further detail on baseline sleep quality discussed"
     - "Not reported this session" (the "this session" hedge)
     - "No specifics provided"
     - "Not addressed in detail"
   EXCEPTION — a "Denies X" / "No X reported" entry is allowed only when it
   reflects a clinical screen the clinician actually conducted (suicidal ideation
   screen, substance-use review, side-effects review). It must be a stand-alone
   clinical finding with no meta-hedge attached.
   MIXED CONTENT — if a field combines a real finding with a meta-comment, KEEP
   the finding and DROP the meta-comment. If nothing remains, omit the field.

R2. FORBIDDEN QUOTES.
   Verbatim quotes are allowed only for these four categories:
     a) statements about suicidal/self-harm ideation, plan, intent, or means;
     b) any disclosure of abuse (historical or current) — preserve exactly;
     c) a patient-stated rating (e.g. "6 out of 10");
     d) a clinician-stated risk level or clinician's risk denial.
   Any quote outside these four is a violation. Paraphrase it. Hard cap: at
   most ONE verbatim quote per section, even from the allowed categories.

R3. VISUAL CUES — AUDIO-ONLY.
   The transcript is audio. Anything that requires SEEING the patient must not
   appear: Appearance, Behaviour (visual), Eye contact, Psychomotor activity,
   AFFECT (quality/range/intensity/reactivity/congruence — e.g. "grief-like
   affect", "anxious affect", "blunted affect", "tearful" as an observation).
   If you see any of these — REMOVE them. Mood (patient-reported feelings),
   Speech (audible features), Thought, and Perception (from what was said) are fine.

R4. SIDE EFFECTS PLACEMENT (Follow-Up Psych specifically, but applies generally).
   The Side Effects field is for genuine medication side effects only. A
   patient's worry that an unrelated event (e.g. a vaccine, an injection)
   caused symptoms is NOT a side effect — it belongs in Stressors/Triggers or
   in Interventions as a misattribution the clinician addressed. If Side
   Effects contains vaccine/injection/non-medication content, REMOVE that
   content from Side Effects. If nothing remains, omit the field.

R5. TANGENTIAL EVIDENCE IN TARGET-SYMPTOM FIELDS.
   Sleep / Appetite / Energy / Concentration / Motivation fields must contain
   CURRENT-SESSION evidence ABOUT that symptom. A night-dose taken during a
   panic episode is medication context, NOT sleep evidence. A febrile episode
   weakness is NOT current energy. A past presentation quote ("when I first
   came, my mental strength was very poor") is NOT current concentration.
   Remove tangential content. If nothing remains, OMIT the field entirely —
   never include the field with a "this is past, not current" disclaimer.

R6. FORMULATION CONTAINMENT.
   The clinician's interpretation, pattern recognition over time, personality/
   trait attribution, "consistent with X ego-structure", "egosyntonic Y traits"
   — this is FORMULATION. It lives in EXACTLY ONE place: the formulation field
   (Clinician's Impression / Assessment). It must NOT appear in: Interpersonal
   Relationships, MSE Insight, MSE Judgment, target symptom fields, or any
   current-status field. If you see formulation leaking elsewhere, REMOVE it
   from that field. The formulation field itself is the place for it.

R7. SECTION RENUMBERING.
   Numbered sections must be sequential 1, 2, 3, … with NO GAPS. If a section
   was omitted because it had no content, renumber the remaining ones.
   "1, 2, 3, 5, 6" → fix to "1, 2, 3, 4, 5".

R8. MEDICATION NAMING.
   Every medication reference includes the drug name with its dose (e.g.
   "Clomipramine 25 mg"), never dose alone ("the 25 mg medication", "the
   recently added 25 mg"). If you see dose-only, supply the name from
   elsewhere in the note if available; otherwise flag.

R9. ONE FACT, ONE PLACE.
   Each discrete clinical event or behavioural finding has a primary home field.
   It must not be re-described in full in multiple sections. Common duplication
   patterns to check:
     - A panic episode described in Subjective Report AND again in Response to
       Medications or Anxiety Symptoms — keep the full account in Subjective
       Report; remove or reduce to a one-clause reference elsewhere.
     - Reassurance-seeking described in a target-symptom field (e.g. Anxiety
       Symptoms) AND again in MSE Thought Process — Thought Process is the
       primary home; remove from the symptom field.
     - A vaccination/stressor event appearing in full in both Side Effects and
       Stressors — keep in Stressors, remove from Side Effects.
   If the same event appears in detail in two or more fields, KEEP the full
   account in its primary home and REMOVE the duplicate (or reduce to a brief
   cross-reference only if genuinely needed).

R10. RISK LEVEL CLINICIAN-ASSIGNED ONLY.
   Risk Level labels (Low / Moderate / High) may appear ONLY if the clinician
   spoke that exact word in the transcript. If the model wrote a level the
   clinician did not state, REMOVE the level label. Lead the field with the
   clinician's actual statement(s) instead.

R11. SPEECH FIELD MISUSE.
   The MSE Speech field is for audible features only: rate, tone, volume,
   quantity. Reassurance-seeking, perseveration, repetitive questioning,
   circumstantiality, magical thinking are THOUGHT findings (Process or
   Content) — never Speech. If Speech contains such content, REMOVE it
   from Speech (it almost certainly already appears under Thought Process;
   if not, move it there). If no audible feature remains, omit Speech.

R12. RESPONSE-TO-MEDICATIONS LABEL CLINICIAN-ASSIGNED ONLY.
   Response category labels (Good response / Partial response / Minimal response /
   Worsening symptoms / Unable to assess) may appear ONLY if the clinician spoke that
   exact term in the transcript. If the model prefixed the field with a label the clinician
   did not state, REMOVE the label and lead with the paraphrase of the patient's report
   instead.

R13. TEMPORAL PRECISION — NO FUTURE CONTENT IN CURRENT-STATUS FIELDS.
   Current-status and functioning fields (Academic/Occupational, Social Functioning,
   Day-to-Day Functioning, target-symptom fields) must contain only what is true NOW
   (this session / since last contact). Future plans (upcoming travel, planned
   procedures, scheduled appointments) must appear ONLY in the Plan section or, if
   they are a stressor the patient is anticipating, in Stressors / Triggers.
   If a current-status field contains a future plan, MOVE the plan content to the
   Plan section or Stressors and remove it from the current-status field. If nothing
   current remains, omit the field.

</rules>

<corrections>
For each violation you find and fix, add an entry to `corrections` with:
  - rule: the rule id from the list above (e.g. "empty_field_disclaimer")
  - location: field or section affected (e.g. "§3 Sleep")
  - detail: one sentence describing what you did

If the note is already clean, return it unchanged with an empty corrections list.
</corrections>

<discipline>
Do NOT invent content. Do NOT add new clinical observations. Do NOT add new
quotes. You may only REMOVE rule-violating content, REPHRASE a quote into
paraphrase, RENUMBER sections, or MOVE content's wording within a sentence to
remove a disclaimer. If a rule says "if nothing remains, omit the field" — omit
the entire line, not just the value.

Preserve everything compliant. Stay quiet on style preferences. Only enforce the
rules above.
</discipline>
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Default to Sonnet 4.6 — same tier as the generator. Callers should pass the
# generator's model name (or a stronger one) for best results.
DEFAULT_MODEL = "claude-sonnet-4-6"


def validate_summary(
    summary: str,
    *,
    session_type: str | None = None,
    model_name: str = DEFAULT_MODEL,
    timeout: int = 240,
    max_output_tokens: int = 32000,
) -> ValidationResult:
    """Run the LLM validator and return a revised summary plus list of corrections.

    Pass the generator's `model_name` so the validator runs at the same (or
    stronger) capability tier. Falls back to Sonnet 4.6 if omitted.
    """

    chat = ChatAnthropic(
        model=model_name,
        max_tokens=max_output_tokens,
        timeout=timeout,
        max_retries=2,
        temperature=0,
    )
    structured = chat.with_structured_output(_ValidatorOutput)

    user_block = (
        f"<session_type>{session_type}</session_type>\n\n" if session_type else ""
    ) + f"<generated_note>\n{summary}\n</generated_note>"

    response = structured.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_block),
        ]
    )
    if not isinstance(response, _ValidatorOutput):
        response = _ValidatorOutput.model_validate(response)

    issues = [
        ValidationIssue(rule=c.rule, location=c.location, detail=c.detail)
        for c in response.corrections
    ]
    return ValidationResult(revised=response.revised_summary, issues=issues)
