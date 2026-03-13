---
name: rf-security-submission-ready
description: >
  Run the submission-ready RF security experiment pipeline (Oracle vs AMC, CRC collapse, efficiency vs noise/jamming, transferability).
argument-hint: "[optional: snr=18 mods=QPSK attack=CW]"
disable-model-invocation: true
---

You are running a security paper evaluation pipeline for control-plane adversarial attacks on adaptive wireless receivers.

## What to load
- The full experiment specification is in ./skill-spec.yaml (same directory).
- Treat it as the authoritative pipeline with inputs/outputs/steps.

## How to run
1) Read skill-spec.yaml and summarize the resolved plan (inputs, outputs, steps).
2) Ask for missing required paths if not provided:
   - dataset_root
   - awn_ckpt
   - run_dir (optional; default to ./results/runs/<timestamp>)
3) Execute steps sequentially. Log progress and save outputs exactly as listed in skill-spec.yaml.

## Hard requirements
- Always separate Oracle and AMC branches; do not conflate AMC accuracy with link reliability.
- Enforce Clean+Oracle sanity gate before claiming attack impact.
- Produce paper-ready artifacts: tables + figures + security-style draft text files.

ARGUMENTS: $ARGUMENTS
