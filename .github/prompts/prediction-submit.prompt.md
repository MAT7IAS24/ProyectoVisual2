# Enviar predicciones del reto Pulso TransMi

Usa la API key desde la variable de entorno `PULSO_API_KEY`.
No la escribas ni la guardes en código.

Haz lo siguiente:
1. Consulta la ruta `GET /v1/forecast-cycles/current`.
2. Lee el `cycle_id` y `data_cutoff` del ciclo activo.
3. Carga el modelo entrenado y genera las predicciones para cada estación objetivo.
4. Construye un payload con `schema_version: "1.0"`.
5. Agrega `Authorization: Bearer <PULSO_API_KEY>` y `Idempotency-Key`.
6. Envía el JSON a `POST /v1/submissions`.
7. Verifica que la respuesta tenga `status: "accepted"` o al menos `201`.
8. Si falla, revisa la documentación OpenAPI y corrige el esquema antes de reintentar.

Usa siempre una `client_run_id` única y un `model.version` con el nombre del experimento, por ejemplo `exp_01_linear_regression`.
