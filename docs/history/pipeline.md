# Pipeline

The authoritative design document is **`.local/docs/pipeline.html`**
(published as an artifact) — the whole system as built on 2026-08-14, with
the measurement behind every design choice: the three scheduled jobs, the
data layer, the release stages, both agents' architectures, the guard
system, the measurement discipline, and the failure stance.

Quick reference:

* Hourly `KaggricultureSameDay` (:35): delta fetch + ingest + collapse
  alarm; 02:35 adds archive/breadth mines + the pre-ranker recall probe.
* Daily `KaggricultureAutopilot` (04:00, ~10 min): fetch warm-up.
* Daily `KaggricultureRefreshCycle` (04:30, ~3-4 h): detect → games → mine →
  retrains (incl. GRU) → Rust parity → tapes (+holdout) → tournament
  (funnel when its gate is open, halving, 4-seed finals) → crown →
  build + graphs + opening exposure → gates → **sha-verified publish**.

Forward plan: `.local/docs/track-p.html`. Chronology: `BUILD_JOURNAL.md`.
Every measurement: `docs/history/issues-and-improvements.md`.
