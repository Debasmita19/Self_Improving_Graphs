#!/usr/bin/env python
"""
demo_drift.py
~~~~~~~~~~~~~
End-to-end demonstration of the project_drift pipeline.

Three scenarios are compared:
  1. **No drift** — reference and current come from the same distribution
     (clinical / EHR prompts).
  2. **Mild drift** — current distribution is a mix of clinical and
     software-engineering prompts.
  3. **Strong drift** — current distribution is entirely software-
     engineering prompts, unrelated to the clinical reference.

Run::

    python -m project_drift.examples.demo_drift
"""

from __future__ import annotations

import random

from project_drift import (
    DriftConfig,
    RuntimeEvent,
    ContextDriftDetector,
    WindowManager,
    should_downweight_archive_branch,
    compute_archive_staleness_weight,
)

# =====================================================================
# Synthetic data generators
# =====================================================================

CLINICAL_PROMPTS = [
    "What is the 30-day mortality rate for patients admitted with acute myocardial infarction?",
    "Retrieve lab results for patient cohort with sepsis diagnosis in the ICU.",
    "Compare length-of-stay distributions between surgical and medical admissions.",
    "Identify patients with chronic kidney disease stage 3 or higher from the EHR.",
    "What medications were prescribed most frequently for heart failure patients?",
    "Calculate the readmission rate within 90 days for pneumonia discharges.",
    "Extract all hemoglobin A1c values for diabetic patients over the past year.",
    "Which antibiotic regimens were associated with lower C. difficile infection rates?",
    "Summarise the vital-sign trends for ICU patients diagnosed with ARDS.",
    "Determine the average time from ED arrival to first antibiotic for sepsis cases.",
    "List comorbidities associated with prolonged mechanical ventilation.",
    "Identify the top-5 adverse drug events reported in the cardiology ward.",
    "How does the SOFA score trajectory differ between survivors and non-survivors?",
    "Retrieve the discharge summaries for patients with acute pancreatitis.",
    "What is the incidence of ventilator-associated pneumonia this quarter?",
]

CLINICAL_CONTEXTS = [
    "Patient demographics table joined with admissions and diagnoses.",
    "Lab events filtered by itemid for troponin and BNP levels.",
    "Procedures table linked to ICD-10 codes for cardiac catheterisation.",
    "Prescriptions table with NDC codes for beta-blockers and ACE inhibitors.",
    "ICU stays table merged with chart events for hourly vital signs.",
]

SOFTWARE_PROMPTS = [
    "Refactor the authentication middleware to use JWT with refresh tokens.",
    "Write unit tests for the payment processing service using pytest.",
    "Optimise the database query that joins orders with inventory tables.",
    "Add a CI/CD pipeline using GitHub Actions for the frontend repository.",
    "Implement a rate limiter middleware for the REST API endpoints.",
    "Debug the memory leak in the WebSocket connection handler.",
    "Migrate the monolithic backend to a microservices architecture.",
    "Set up Kubernetes deployment manifests for the staging environment.",
    "Profile and optimise the image processing pipeline for lower latency.",
    "Implement OAuth2 PKCE flow for the single-page application.",
    "Add Prometheus metrics and Grafana dashboards for service monitoring.",
    "Write a data migration script for the PostgreSQL schema upgrade.",
    "Implement a message queue consumer for asynchronous order processing.",
    "Fix the race condition in the concurrent file-upload handler.",
    "Design a caching layer using Redis for frequently accessed user profiles.",
]

SOFTWARE_CONTEXTS = [
    "Git diff of src/auth/middleware.ts showing added JWT verification.",
    "Docker Compose configuration for the local development stack.",
    "Terraform module outputs for the AWS ECS cluster.",
    "OpenAPI 3.0 specification for the /api/v2/orders endpoint.",
    "Load test results from k6 showing p99 latency at 500 RPS.",
]


def _make_event(prompt: str, context: str, branch: str = "main") -> RuntimeEvent:
    return RuntimeEvent(
        prompt_text=prompt,
        retrieved_context=context,
        metadata={"source": "demo"},
        archive_branch_id=branch,
    )


def _sample_events(
    prompts: list[str],
    contexts: list[str],
    n: int,
    branch: str = "main",
) -> list[RuntimeEvent]:
    return [
        _make_event(
            random.choice(prompts),
            random.choice(contexts),
            branch=branch,
        )
        for _ in range(n)
    ]


# =====================================================================
# Main demo
# =====================================================================

def main() -> None:
    random.seed(42)

    cfg = DriftConfig(
        reference_window_size=60,
        current_window_size=30,
        p_val_threshold=0.05,
        n_permutations=200,
    )

    print("=" * 70)
    print("  project_drift — Runtime Context Drift Detection Demo")
    print("=" * 70)

    # --- build reference from clinical prompts ---
    wm = WindowManager(cfg)
    ref_events = _sample_events(CLINICAL_PROMPTS, CLINICAL_CONTEXTS, 60, branch="archive-v1")
    wm.add_reference_events(ref_events)
    wm.freeze_reference()

    detector = ContextDriftDetector(cfg)
    detector.fit(wm.reference_events)

    # --- Scenario 1: no drift ---
    print("\n--- Scenario 1: No drift (same clinical distribution) ---")
    cur_same = _sample_events(CLINICAL_PROMPTS, CLINICAL_CONTEXTS, 30, branch="archive-v1")
    report_1 = detector.test(cur_same, notes="same distribution")
    print(report_1.summary_line())
    print(f"  Downweight branch?  {should_downweight_archive_branch(report_1, cfg)}")
    print(f"  Staleness weight:   {compute_archive_staleness_weight(report_1, cfg):.4f}")

    # --- Scenario 2: mild drift (mixed) ---
    print("\n--- Scenario 2: Mild drift (50/50 clinical + software) ---")
    cur_mixed = (
        _sample_events(CLINICAL_PROMPTS, CLINICAL_CONTEXTS, 15, branch="archive-v2")
        + _sample_events(SOFTWARE_PROMPTS, SOFTWARE_CONTEXTS, 15, branch="archive-v2")
    )
    random.shuffle(cur_mixed)
    report_2 = detector.test(cur_mixed, notes="mixed distribution")
    print(report_2.summary_line())
    print(f"  Downweight branch?  {should_downweight_archive_branch(report_2, cfg)}")
    print(f"  Staleness weight:   {compute_archive_staleness_weight(report_2, cfg):.4f}")

    # --- Scenario 3: strong drift ---
    print("\n--- Scenario 3: Strong drift (entirely software prompts) ---")
    cur_shifted = _sample_events(SOFTWARE_PROMPTS, SOFTWARE_CONTEXTS, 30, branch="archive-v3")
    report_3 = detector.test(cur_shifted, notes="shifted distribution")
    print(report_3.summary_line())
    print(f"  Downweight branch?  {should_downweight_archive_branch(report_3, cfg)}")
    print(f"  Staleness weight:   {compute_archive_staleness_weight(report_3, cfg):.4f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Scenario 1 (no drift):     score = {report_1.drift_score:.4f}  →  drift = {report_1.drift_detected}")
    print(f"  Scenario 2 (mild drift):   score = {report_2.drift_score:.4f}  →  drift = {report_2.drift_detected}")
    print(f"  Scenario 3 (strong drift): score = {report_3.drift_score:.4f}  →  drift = {report_3.drift_detected}")
    print()

    # Sanity: strong drift should produce higher score than no-drift
    assert report_3.drift_score > report_1.drift_score, (
        "Expected strong-drift score > no-drift score"
    )
    print("  ✓  Drift scores are ordered as expected.")
    print()


if __name__ == "__main__":
    main()
