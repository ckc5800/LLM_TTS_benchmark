# tts-benchmark gRPC Bridge

`d:/tts-benchmark` models can be exposed as AICESS-compatible gRPC endpoints with this bridge.

## 1) What this does

- Uses `adapters/run_model.py` as subprocess backend.
- Exposes `tts.v1.TTSService` methods used by admin/api.
- Returns stub responses for unsupported features (batch/backup/add/remove voice).

## 2) Prerequisites

- Python env with `grpcio` installed (server runtime env).
- AICESS proto python files must exist in one of:
  - `d:/aicess-tts-integrated/admin/backend/app/grpc_client/proto`
  - `d:/aicess-tts-integrated/api/proto`
- Benchmark model/env paths configured in `grpc_bridge/models.json`.

## 3) Run

```powershell
cd d:\tts-benchmark
python grpc_bridge\benchmark_grpc_server.py --port 8320
```

Optional path overrides:

```powershell
python grpc_bridge\benchmark_grpc_server.py `
  --models-root D:\models `
  --py-list-root D:\list `
  --port 8320
```

If proto path is custom:

```powershell
$env:AICESS_TTS_PROTO_PY_DIR="D:\somewhere\grpc_client"
python grpc_bridge\benchmark_grpc_server.py --port 8320
```

## 4) Notes

- This bridge is subprocess-based, so per-request latency is higher than in-process engines.
- `models.json` controls voice/model mapping and python executables.
- Only WAV synthesis is supported for now.
