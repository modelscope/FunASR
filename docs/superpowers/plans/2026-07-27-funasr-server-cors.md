# Trusted Browser CORS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit, default-disabled trusted browser origins to `funasr-server` so local web applications can consume successful OpenAI-compatible transcription responses.

**Architecture:** The CLI collects repeated exact origins and passes them into `create_app`. The app normalizes the list and conditionally installs FastAPI/Starlette `CORSMiddleware`; transcription routes and model loading remain unchanged.

**Tech Stack:** Python, argparse, FastAPI, Starlette CORS middleware, pytest, curl.

## Global Constraints

- No CORS middleware is installed unless at least one non-empty origin is supplied.
- Allowed methods are exactly `GET`, `POST`, and `OPTIONS`.
- Allowed headers are exactly `Authorization` and `Content-Type`.
- `allow_credentials` remains `False`.
- Existing model loading and transcription response behavior must not change.
- Every behavior change follows red-green TDD.

---

### Task 1: Conditional CORS Middleware

**Files:**
- Modify: `tests/test_server_app_openai_segments.py`
- Modify: `funasr/bin/_server_app.py`

**Interfaces:**
- Consumes: optional `cors_origins` iterable supplied by callers
- Produces: `create_app(..., cors_origins=None) -> FastAPI` with conditional middleware

- [ ] **Step 1: Extend the FastAPI test stub and write failing tests**

Add middleware capture to `DummyFastAPI` and stub `fastapi.middleware.cors.CORSMiddleware`:

```python
class DummyFastAPI:
    def __init__(self, *args, **kwargs):
        self.state = types.SimpleNamespace()
        self.routes = {}
        self.metadata = kwargs
        self.middleware = []

    def add_middleware(self, middleware_class, **kwargs):
        self.middleware.append((middleware_class, kwargs))

class DummyCORSMiddleware:
    pass
```

Register the middleware stubs in `sys.modules`, then add behavior tests with hand-derived literal expectations:

```python
def test_server_cors_is_disabled_by_default(monkeypatch):
    module = load_server_app(monkeypatch)
    install_dummy_funasr(monkeypatch)

    app = module.create_app(device="cpu", preload_model="sensevoice")

    assert app.middleware == []


def test_server_configures_normalized_trusted_origins(monkeypatch):
    module = load_server_app(monkeypatch)
    install_dummy_funasr(monkeypatch)

    app = module.create_app(
        device="cpu",
        preload_model="sensevoice",
        cors_origins=[
            " http://localhost:3000 ",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            " ",
        ],
    )

    assert app.middleware == [
        (
            module.CORSMiddleware,
            {
                "allow_origins": [
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                ],
                "allow_credentials": False,
                "allow_methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Authorization", "Content-Type"],
            },
        )
    ]
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_server_app_openai_segments.py -q
```

Expected: the new test fails because `create_app` does not accept `cors_origins`.

- [ ] **Step 3: Implement the minimum middleware behavior**

Import `CORSMiddleware`, add the optional `cors_origins` parameter, normalize with first-seen de-duplication, and call:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=normalized_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same pytest command. Expected: all focused tests pass.

- [ ] **Step 5: Commit the tested middleware**

Stage only the two task files and create a signed commit named `feat(server): allow trusted browser origins`.

### Task 2: Repeatable CLI Option

**Files:**
- Modify: `tests/test_server_app_openai_segments.py`
- Modify: `funasr/bin/server.py`

**Interfaces:**
- Consumes: repeated `--cors-origin ORIGIN` arguments
- Produces: `args.cors_origin: list[str] | None`, passed as `cors_origins=args.cors_origin`

- [ ] **Step 1: Write a failing CLI forwarding test**

Extract the existing parser construction into `build_parser()` and add a parser behavior test:

```python
def test_server_cli_collects_repeated_cors_origins():
    module = load_server_cli()

    args = module.build_parser().parse_args(
        [
            "--cors-origin",
            "http://localhost:3000",
            "--cors-origin",
            "http://127.0.0.1:3000",
        ]
    )

    assert args.cors_origin == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
```

- [ ] **Step 2: Run the single test and verify RED**

Expected: argparse rejects `--cors-origin`.

- [ ] **Step 3: Add the repeatable CLI argument**

Move the current parser setup into `build_parser()`, use the parser from `main()`, and add:

```python
parser.add_argument(
    "--cors-origin",
    action="append",
    default=None,
    metavar="ORIGIN",
    help="Trusted browser origin for CORS; repeat for multiple origins (disabled by default)",
)
```

Pass the parsed list into the real application boundary:

```python
app = create_app(
    device=args.device,
    preload_model=args.model,
    model_path=args.model_path,
    hub=args.hub,
    cors_origins=args.cors_origin,
)
```

- [ ] **Step 4: Run focused and complete server tests**

Expected: CLI forwarding and all server tests pass.

- [ ] **Step 5: Commit the tested CLI surface**

Create a signed commit named `feat(server): expose trusted CORS origins`.

### Task 3: Operator Documentation

**Files:**
- Modify: `docs/troubleshooting.md`
- Modify: `docs/troubleshooting_zh.md`

**Interfaces:**
- Consumes: public `--cors-origin` CLI option
- Produces: bilingual, copy-pasteable browser deployment guidance

- [ ] **Step 1: Add concise English and Chinese guidance**

Document the browser CORS symptom, a trusted-origin startup command, repeated origins, and the requirement to use the browser's exact scheme/host/port.

- [ ] **Step 2: Review the rendered Markdown contract**

Confirm both commands use the same public CLI, both explain exact origins, neither recommends wildcard access, and neither claims CORS is enabled by default.

- [ ] **Step 3: Run docs and server verification**

Run the focused docs tests, all server tests, Black, Ruff, compilation, and `git diff --check`.

- [ ] **Step 4: Commit documentation**

Create a signed commit named `docs(server): explain browser CORS setup`.

### Task 4: Real Browser-Contract Smoke and Publication

**Files:**
- No production file changes unless verification exposes a tested defect.

**Interfaces:**
- Consumes: exact feature branch head
- Produces: live preflight/transcription evidence and a reviewable FunASR PR

- [ ] **Step 1: Start the exact branch server on CPU**

Use SenseVoice and `--cors-origin http://127.0.0.1:3000` on an unused port.

- [ ] **Step 2: Verify matching and non-matching origins**

Assert matching-origin `OPTIONS` returns 200 with the expected allow-origin/method/header values, matching-origin multipart POST returns 200 with transcription text and allow-origin, and an unlisted origin receives no allow-origin header.

- [ ] **Step 3: Verify complete repository gates**

Run relevant tests, formatting, lint, compilation, signature checks, and diff checks at the exact head.

- [ ] **Step 4: Push with rollback protection and open a ready PR**

Push the signed branch, create a non-draft PR with exact test and runtime evidence, wait for repository CI, and merge only if all code-owned gates pass.

- [ ] **Step 5: Refresh NextChat #6860 evidence**

Re-run its exact `transcribeAudio` request against the CORS-enabled server, update the PR body with the required server command and real browser-contract evidence, and route one review request to the active NextChat maintainer.
