# Business Requirements Document
## Problem
Lending teams need a repeatable signal for prioritizing applicants with elevated estimated default risk. The product provides a model-driven risk score/tier for human review.

## Personas
| Persona | Goal | Pain Point |
|---|---|---|
| Loan Officer | Prioritize applications | Manual/inconsistent screening |
| Credit Risk Analyst | Assess model and applicant risk | Need transparent metrics and reproducibility |

## Objectives
1. Train AdaBoost on the specified dataset.
2. Use an 80/20 stratified train-test split.
3. Perform CV/tuning only inside the training partition.
4. Select and document an operating threshold.
5. Serve scoring through FastAPI.
6. Provide a React analyst UI.
7. Publish a model card.

## Requirements
| ID | Requirement | Priority | Acceptance |
|---|---|---|---|
| BR-001 | Ingest dataset | Must | Source/version/schema documented |
| BR-002 | Validate data | Must | Missingness, duplicates, types and ranges checked |
| BR-003 | Perform EDA | Must | Findings and plots documented |
| BR-004 | Establish baseline | Must | Baseline metrics recorded |
| BR-005 | Train AdaBoost | Must | Versioned artifact produced |
| BR-006 | Tune and compare | Must | CV/search process documented |
| BR-007 | Final test evaluation | Must | Test untouched until final evaluation |
| BR-008 | Risk tiers | Must | Threshold/tier mapping documented |
| BR-009 | Scoring API | Must | Validated versioned endpoint |
| BR-010 | React UI | Must | End-to-end scoring works |
| BR-011 | Model card | Must | Use, data, metrics, limits documented |
| BR-012 | Experiment tracking | Should | Run configuration and metrics recorded |

## NFRs
Performance, reliability, security, accessibility, scalability, cost and responsible-use limitations must be documented. Avoid real PII in the student environment.

## Risks
Dataset/business mismatch, class imbalance, leakage, threshold misuse, overfitting, historical bias and distribution shift.
