---
applyTo: "**/*.{py,md,yml,yaml}"
---

# Prediction submission workflow for Pulso TransMi

Use this workflow when the user needs to submit predictions for the Pulso TransMi challenge.

## Critical rules
- Never hardcode or paste the API key into source files.
- Read the key from the environment variable `PULSO_API_KEY`.
- If needed, also read `PULSO_API_URL`; default is `https://pulso-transmi.72-60-245-2.sslip.io`.
- Use the Bearer token format: `Authorization: Bearer <PULSO_API_KEY>`.
- Always include the `Idempotency-Key` header on every `/v1/submissions` request.
- Validate the current cycle before building the payload: `GET /v1/forecast-cycles/current`.
- Use `schema_version: "1.0"`.
- Use the cycle `cycle_id` and `data_cutoff` exactly as returned by the API.
- Each prediction item must include:
  - `station_id` as a 5-digit string
  - `target_at` ISO-8601 UTC datetime
  - `value` numeric demand prediction
- Set `model.version` to the experiment tag, for example `exp_01_linear_regression`.
- Include `trained_at`, `training_data_end`, and if available `git_commit`.
- Submit to `POST /v1/submissions`.
- Treat `201` as success, and verify `status` is `accepted` in the response body.
- If the API rejects the payload, inspect both the request body and the OpenAPI contract before retrying.

## Recommended sequence
1. Read the active cycle:
   `GET /v1/forecast-cycles/current`
2. Load the trained model artifact and feature metadata.
3. Generate predictions for each target in the cycle.
4. Build the JSON payload:
   - `schema_version`
   - `cycle_id`
   - `client_run_id`
   - `data_cutoff`
   - `model`
   - `predictions`
5. Send it with `Authorization` and `Idempotency-Key` headers.
6. Print the response and confirm it was accepted.

## Good defaults for a script
- `client_run_id`: `f"exp_01_{uuid.uuid4().hex[:12]}"`
- `trained_at`: current UTC timestamp
- `training_data_end`: cycle `data_cutoff`
- `git_commit`: optional, from `git rev-parse HEAD`

## Example request shape
```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Idempotency-Key": uuid.uuid4().hex,
}

payload = {
    "schema_version": "1.0",
    "cycle_id": cycle["cycle_id"],
    "client_run_id": "exp_01_abc123",
    "data_cutoff": cycle["data_cutoff"],
    "model": {
        "version": "exp_01_linear_regression",
        "trained_at": "2026-09-18T20:00:00Z",
        "training_data_end": cycle["data_cutoff"],
        "git_commit": "<commit-hash>"
    },
    "predictions": [
        {"station_id": "02300", "target_at": "2026-09-09T05:00:00Z", "value": 120.1098}
    ]
}
```

## Safety notes
- Do not save the key in the repo, `.env` if it is tracked by git, or a notebook output.
- Prefer environment variables or GitHub Actions secrets.
- Treat every submission as idempotent by generating a fresh `Idempotency-Key`.
