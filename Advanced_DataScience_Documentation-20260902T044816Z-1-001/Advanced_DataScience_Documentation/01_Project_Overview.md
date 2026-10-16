# Micro-Loan Risk Scorer — Project Overview
**Track:** Advanced Data Science | **Domain:** FinTech/Lending | **Duration:** 12 weeks | **Team:** 2 | **Level:** Intermediate

## Goal
Build an AdaBoost-based credit-risk scorer that flags high-default-risk microloan applicants and exposes risk-tier scoring through a FastAPI service with a React frontend.

## Inputs and Deliverables
- Give Me Some Credit dataset, as specified in the project brief.
- Research grounding: *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting*, Journal of Computer and System Sciences, 1997.
- Required split: 80/20 stratified train-test.
- Required outputs: trained AdaBoost model, risk-tier scoring API, model card.
- Required evaluation: ROC-AUC and precision/recall at a chosen threshold.
- Stack: Python, JavaScript, scikit-learn, FastAPI, React, pandas, NumPy, matplotlib, Tailwind CSS, Axios.

## Core Lifecycle
Data → Cleaning → EDA → Features → Training → Validation → Evaluation → Deployment.

## MVP
Dataset validation, EDA, feature engineering, baseline, AdaBoost, tuning, cross-validation, test evaluation, threshold/risk tiers, FastAPI, React, model card, tests and deployment.

## Important Assumption
The exact dataset schema, target definition, row count, feature meanings and class distribution must be taken from the actual dataset/documentation. They are not fabricated here. The “micro-loan” business context is an intended application context and must be assessed for dataset fit.

## Product Boundary
This is a decision-support prototype, not an autonomous loan-approval system.
