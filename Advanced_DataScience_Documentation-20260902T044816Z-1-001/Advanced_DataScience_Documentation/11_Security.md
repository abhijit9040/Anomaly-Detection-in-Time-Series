# Security Design
Authentication/authorization should be added when the API is deployed beyond local development. Use roles such as LOAN_OFFICER, CREDIT_RISK_ANALYST and ADMIN if role separation is needed.

## Controls
- FastAPI/Pydantic input validation.
- HTTPS.
- Rate limiting for public endpoints.
- CORS restrictions.
- Secure secrets.
- Dependency scanning.
- Secure logging with no raw applicant records.
- No real PII for the student demo.
- Protect and integrity-check serialized model artifacts.
- Pin dependencies.

## ML Security
Consider data poisoning, leakage, distribution shift, adversarial input manipulation, historical bias and threshold misuse.

## Responsible Use
The score is not a guarantee of default and must not be presented as an autonomous lending decision.
