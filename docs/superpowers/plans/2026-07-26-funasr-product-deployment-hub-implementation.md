# FunASR Product Deployment Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a bilingual, evidence-backed deployment product center on `www.funasr.com` that guides users to vLLM, llama.cpp/GGUF, OpenAI-compatible API, real-time, container, and CPU production paths.

**Architecture:** A Python 3.11 static-site builder reads one bilingual JSON deployment registry, renders Jinja2 templates, copies a tracked legacy-site snapshot, and validates every generated route before publishing. The production host receives immutable versioned releases; Nginx switches an atomic `current` symlink, records fixed outbound conversions, and can roll back to the prior release without rebuilding.

**Tech Stack:** Python 3.11, Jinja2 3.1.6, Beautiful Soup 4.13.4, pytest 8.4.1, dependency-free HTML/CSS/JavaScript, bundled Lucide SVG assets, Nginx, Playwright.

## Global Constraints

- The site is bilingual Chinese and English; every new canonical route has a language pair with identical structure.
- Production claims require a verification date, tested version, evidence URL, smoke test, and explicit limitation.
- Existing indexed routes remain available; `/llama-cpp.html` and `/en/llama-cpp.html` remain permanent compatibility entries.
- The live server remains Nginx-only and receives complete static output, never a Python runtime.
- Generated assets use content-hashed filenames; HTML is no-cache and immutable assets are cached long term.
- Primary navigation order is Product, Deploy, Models, Benchmarks, Ecosystem, Blog, Docs, Donors; Donors remains last.
- Use white and near-black foundations, cobalt actions, green verified state, amber limitations, 4-8 px corners, zero negative letter spacing, and no decorative gradients or blobs.
- The hero uses a custom bitmap with no generated text; terminal and API output remains inspectable HTML.
- No public inference API, authentication layer, rate limiter, or Prometheus service is added in this website release.
- Every production deployment creates a new immutable release directory and retains the previous release for rollback.

---

## File Structure

```text
web-pages/product-site/
  assets/
    css/site.css
    images/deployment-topology.webp
    js/site.js
    lucide/
  content/
    legacy-manifest.json
  data/
    deployments.json
    navigation.json
  legacy/
  templates/
    base.html
    home.html
    deploy-index.html
    deploy-detail.html
    benchmarks.html
    404.html
  tests/
    test_build.py
    test_registry.py
    test_selector.py
    test_legacy.py
    test_output.py
  build.py
  registry.py
  selector.py
  legacy.py
  validate.py
  requirements-site.txt
web-pages/scripts/
  import-live-site.sh
  deploy-product-site.sh
  rollback-product-site.sh
web-pages/nginx/
  funasr.com.conf
  conversion-map.conf
docs/superpowers/plans/2026-07-26-funasr-product-deployment-hub-implementation.md
```

`registry.py` owns schema and evidence validation. `selector.py` owns deterministic recommendations. `legacy.py` owns parser-based navigation normalization. `build.py` composes those modules and writes output. `validate.py` checks a complete output directory without changing it. Templates contain page structure only; data and claims live in `deployments.json`.

---

### Task 1: Static Builder Contract and Registry Validation

**Files:**
- Create: `web-pages/product-site/requirements-site.txt`
- Create: `web-pages/product-site/registry.py`
- Create: `web-pages/product-site/data/deployments.json`
- Create: `web-pages/product-site/data/navigation.json`
- Test: `web-pages/product-site/tests/test_registry.py`

**Interfaces:**
- Produces: `load_registry(path: Path) -> dict[str, object]`
- Produces: `validate_registry(data: dict[str, object]) -> list[str]`
- Produces: `deployment_pairs(data: dict[str, object]) -> list[tuple[dict, dict]]`
- Registry entry ids: `vllm`, `llama-cpp`, `openai-api`, `realtime`, `containers`, `cpu-runtime`, `production`

- [ ] **Step 1: Pin build dependencies**

```text
Jinja2==3.1.6
beautifulsoup4==4.13.4
pytest==8.4.1
```

- [ ] **Step 2: Write failing registry tests**

```python
def test_production_entry_requires_evidence():
    data = valid_registry()
    del data['deployments'][0]['evidence']
    assert validate_registry(data) == ['vllm: production-verified entry requires evidence']

def test_language_pairs_have_identical_fields():
    data = load_registry(REGISTRY)
    assert validate_registry(data) == []
    assert all(set(zh) == set(en) for zh, en in deployment_pairs(data))

def test_registry_has_all_product_routes():
    data = load_registry(REGISTRY)
    assert {item['id'] for item in data['deployments']} == {
        'vllm', 'llama-cpp', 'openai-api', 'realtime',
        'containers', 'cpu-runtime', 'production',
    }
```

- [ ] **Step 3: Run the tests and verify the missing-module failure**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_registry.py -q`

Expected: FAIL because `registry.py` and `deployments.json` do not exist.

- [ ] **Step 4: Implement strict registry validation**

`validate_registry` must reject duplicate ids/routes, missing language fields, invalid maturity values, missing production evidence, malformed `https://` evidence URLs, missing limitations, and benchmark records without model/runtime/hardware/workload/settings/source/verified date. Return stable `"<entry id>: <reason>"` messages rather than raising on content errors.

- [ ] **Step 5: Add evidence-backed registry records**

Populate all seven entries from repository docs and releases. Each command, alias, tested version, and benchmark source must resolve to a tracked FunASR document or GitHub release. Mark entries `community-verified` when a complete production verification record is unavailable.

- [ ] **Step 6: Run registry tests**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_registry.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the registry contract**

```bash
git add web-pages/product-site/requirements-site.txt web-pages/product-site/registry.py web-pages/product-site/data web-pages/product-site/tests/test_registry.py
git commit -S -m "feat(site): add deployment evidence registry"
```

### Task 2: Deterministic Deployment Selector

**Files:**
- Create: `web-pages/product-site/selector.py`
- Test: `web-pages/product-site/tests/test_selector.py`

**Interfaces:**
- Consumes: validated deployment entries from `load_registry`
- Produces: `recommend(entries: list[dict], workload: str, hardware: str, priority: str) -> dict`
- Accepted workload values: `batch`, `realtime`, `private-api`, `edge`
- Accepted hardware values: `nvidia-gpu`, `cpu`, `desktop-edge-gpu`, `kubernetes`
- Accepted priority values: `throughput`, `latency`, `portability`, `compatibility`

- [ ] **Step 1: Write recommendation tests**

```python
@pytest.mark.parametrize(('workload', 'hardware', 'priority', 'expected'), [
    ('batch', 'nvidia-gpu', 'throughput', 'vllm'),
    ('edge', 'cpu', 'portability', 'llama-cpp'),
    ('private-api', 'kubernetes', 'compatibility', 'containers'),
    ('realtime', 'nvidia-gpu', 'latency', 'realtime'),
])
def test_recommendation_matrix(workload, hardware, priority, expected, entries):
    assert recommend(entries, workload, hardware, priority)['id'] == expected

def test_unknown_selector_value_is_rejected(entries):
    with pytest.raises(ValueError, match='Unsupported hardware: tpu'):
        recommend(entries, 'batch', 'tpu', 'throughput')
```

- [ ] **Step 2: Verify tests fail**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_selector.py -q`

Expected: FAIL because `selector.recommend` does not exist.

- [ ] **Step 3: Implement table-driven scoring**

Score only exact registry capabilities. Break ties by the registry's numeric `selector_rank`, then stable id. Return the full winning entry plus `selection_reason` and `primary_limitation`; never infer unsupported hardware or maturity.

- [ ] **Step 4: Run selector tests**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_selector.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web-pages/product-site/selector.py web-pages/product-site/tests/test_selector.py
git commit -S -m "feat(site): add deployment recommendation engine"
```

### Task 3: Reproducible Legacy Snapshot

**Files:**
- Create: `web-pages/scripts/import-live-site.sh`
- Create: `web-pages/product-site/content/legacy-manifest.json`
- Create: `web-pages/product-site/legacy/`
- Test: `web-pages/product-site/tests/test_legacy.py`

**Interfaces:**
- Consumes: public files under `funasr-web:/root/FunASR/web-pages/dist`
- Produces: tracked `legacy/` corpus and SHA-256 `legacy-manifest.json`
- Excludes: backups, access statistics, logs, `.git`, temporary files, and private configuration

- [ ] **Step 1: Write manifest integrity test**

```python
def test_legacy_manifest_matches_snapshot():
    manifest = json.loads(MANIFEST.read_text())
    actual = {str(path.relative_to(LEGACY)): sha256(path) for path in public_files(LEGACY)}
    assert actual == manifest['files']

def test_private_or_generated_files_are_excluded():
    forbidden = {'access.log', 'stats.json', '.git', 'backup'}
    assert not any(part in forbidden for path in public_files(LEGACY) for part in path.parts)
```

- [ ] **Step 2: Verify the empty snapshot fails**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_legacy.py -q`

Expected: FAIL because no manifest exists.

- [ ] **Step 3: Implement allowlist-based import**

The script must copy only HTML, CSS, JavaScript, JSON content, fonts, images, audio, video, favicon, robots, and sitemap files. It writes into a fresh temporary directory, computes hashes, and atomically replaces `legacy/` only after every file passes the allowlist.

- [ ] **Step 4: Import from a read-only production snapshot**

Run the importer from `ind-gpu8` using an archive downloaded from `funasr-web`; do not mutate the production host. Review `git status --short` and the manifest before staging.

- [ ] **Step 5: Run manifest tests**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_legacy.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web-pages/scripts/import-live-site.sh web-pages/product-site/content/legacy-manifest.json web-pages/product-site/legacy web-pages/product-site/tests/test_legacy.py
git commit -S -m "build(site): track reproducible live-site corpus"
```

### Task 4: Templates, Design System, and Static Build

**Files:**
- Create: `web-pages/product-site/build.py`
- Create: `web-pages/product-site/templates/base.html`
- Create: `web-pages/product-site/templates/home.html`
- Create: `web-pages/product-site/templates/deploy-index.html`
- Create: `web-pages/product-site/templates/404.html`
- Create: `web-pages/product-site/assets/css/site.css`
- Create: `web-pages/product-site/assets/js/site.js`
- Create: `web-pages/product-site/assets/images/deployment-topology.webp`
- Create: selected files under `web-pages/product-site/assets/lucide/`
- Test: `web-pages/product-site/tests/test_build.py`

**Interfaces:**
- Consumes: registry, navigation, selector table, templates, assets, and legacy snapshot
- Produces: `build(output_dir: Path) -> dict[str, object]`
- Produces: `dist/deployment-manifest.json` with route, language, canonical, hreflang, and asset hashes

- [ ] **Step 1: Write failing route and metadata tests**

```python
def test_build_emits_bilingual_product_routes(tmp_path):
    manifest = build(tmp_path)
    routes = {page['route'] for page in manifest['pages']}
    assert {'/', '/en/', '/deploy/', '/en/deploy/', '/404.html'} <= routes

def test_home_has_literal_brand_heading_and_next_section(tmp_path):
    build(tmp_path)
    soup = BeautifulSoup((tmp_path / 'index.html').read_text(), 'html.parser')
    assert soup.h1.get_text(strip=True) == 'FunASR'
    assert soup.select_one('[data-section="deployment-selector"]')
```

- [ ] **Step 2: Verify tests fail**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_build.py -q`

Expected: FAIL because the builder and templates do not exist.

- [ ] **Step 3: Generate the hero bitmap**

Create one 2400x1350 WebP source showing an inspectable deployment topology with GPU servers, CPU/edge devices, and API clients. The image contains no text, logos, gradients, bokeh, or decorative shapes. Record the generation prompt and SHA-256 in `assets/images/README.md`.

- [ ] **Step 4: Vendor only used Lucide icons**

Pin one Lucide release, check in the required SVG files plus its license, and use them for menu, GitHub, external-link, copy, terminal, server, CPU, GPU, shield, activity, and language actions.

- [ ] **Step 5: Implement shared templates and hashed assets**

The base template emits canonical, hreflang, description, Open Graph, structured data, skip link, semantic header/main/footer, and the exact navigation order. `build.py` hashes CSS/JS/image assets, renders into a new directory, and only replaces the requested output after a successful render.

- [ ] **Step 6: Implement home and deployment selector UI**

Render the hero, selector segmented controls, no-JavaScript deployment matrix, proof band, API contract, production-responsibility table, verified downloads, ecosystem references, and final deployment/GitHub actions. `site.js` mirrors the Python selector table embedded as JSON and updates one fixed-size result region without layout shift.

- [ ] **Step 7: Run build tests and inspect generated HTML**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_build.py web-pages/product-site/tests/test_selector.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add web-pages/product-site/build.py web-pages/product-site/templates web-pages/product-site/assets web-pages/product-site/tests/test_build.py
git commit -S -m "feat(site): build bilingual deployment product surface"
```

### Task 5: Deployment Detail and Benchmark Pages

**Files:**
- Create: `web-pages/product-site/templates/deploy-detail.html`
- Create: `web-pages/product-site/templates/benchmarks.html`
- Modify: `web-pages/product-site/build.py`
- Modify: `web-pages/product-site/assets/css/site.css`
- Modify: `web-pages/product-site/assets/js/site.js`
- Test: `web-pages/product-site/tests/test_output.py`

**Interfaces:**
- Consumes: seven validated registry entries and their benchmark records
- Produces: seven Chinese routes, seven English routes, and bilingual benchmark routes

- [ ] **Step 1: Write generated-output tests**

```python
def test_every_deployment_page_has_operational_contract(built_site):
    for page in deployment_pages(built_site):
        soup = BeautifulSoup(page.read_text(), 'html.parser')
        assert soup.select_one('[data-field="verified-date"]')
        assert soup.select_one('[data-section="fit"]')
        assert soup.select_one('[data-section="commands"]')
        assert soup.select_one('[data-section="smoke-test"]')
        assert soup.select_one('[data-section="security"]')
        assert soup.select_one('[data-section="limitations"]')

def test_benchmark_rows_have_complete_conditions(built_site):
    for row in benchmark_rows(built_site):
        assert all(row.get(field) for field in REQUIRED_BENCHMARK_FIELDS)
```

- [ ] **Step 2: Verify tests fail**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_output.py -q`

Expected: FAIL because detail and benchmark pages are not generated.

- [ ] **Step 3: Implement the shared detail template**

Render status/version/evidence, fit and non-fit, support matrix, copy-ready install/start/smoke commands, request flow, capacity variables, security boundary, operations checklist, limitations, troubleshooting, release assets, issue link, and GitHub action from registry data only.

- [ ] **Step 4: Implement benchmark rendering**

Render batch RTFx and real-time capacity in separate tables. Every row includes model, runtime, hardware, data/audio duration, batch or concurrency, warmup/download exclusion, source link, and verification date. Add an explicit warning that results do not transfer across traffic profiles.

- [ ] **Step 5: Run output tests**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_output.py -q`

Expected: PASS for all 18 new bilingual/product routes.

- [ ] **Step 6: Commit**

```bash
git add web-pages/product-site/templates web-pages/product-site/build.py web-pages/product-site/assets web-pages/product-site/tests/test_output.py
git commit -S -m "feat(site): add production deployment guides"
```

### Task 6: Legacy Navigation and Compatibility Routes

**Files:**
- Create: `web-pages/product-site/legacy.py`
- Modify: `web-pages/product-site/build.py`
- Modify: `web-pages/product-site/tests/test_legacy.py`
- Modify: `web-pages/product-site/tests/test_output.py`

**Interfaces:**
- Consumes: legacy HTML and `navigation.json`
- Produces: `normalize_document(html: str, route: str, language: str) -> str`
- Produces: old llama.cpp pages with canonical links to `/deploy/llama-cpp.html` language peers

- [ ] **Step 1: Write parser/idempotency tests**

```python
def test_navigation_normalization_is_idempotent(sample_legacy_html):
    once = normalize_document(sample_legacy_html, '/blog/example.html', 'zh')
    twice = normalize_document(once, '/blog/example.html', 'zh')
    assert once == twice
    assert nav_labels(once)[-1] == '功德榜'

def test_old_llama_route_points_to_product_page(built_site):
    soup = read_soup(built_site / 'llama-cpp.html')
    assert soup.select_one('link[rel="canonical"]')['href'].endswith('/deploy/llama-cpp.html')
```

- [ ] **Step 2: Verify tests fail**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_legacy.py web-pages/product-site/tests/test_output.py -q`

Expected: FAIL because legacy normalization is absent.

- [ ] **Step 3: Implement Beautiful Soup normalization**

Replace only the identified primary navigation container, canonical/hreflang metadata, and versioned shared assets. Preserve article body, embedded media, donor content, demo contracts, and unknown custom markup byte-for-byte outside the normalized nodes. Abort the build when an expected page shell cannot be parsed.

- [ ] **Step 4: Build compatibility pages and verify stable routes**

Keep quickstart, models, ecosystem, blog, donors, demos, and voice assets at existing URLs. Preserve old llama.cpp HTML as useful compatibility content and add canonical/product navigation rather than returning a blank redirect.

- [ ] **Step 5: Run legacy and output tests**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_legacy.py web-pages/product-site/tests/test_output.py -q`

Expected: PASS and a second build produces no output changes.

- [ ] **Step 6: Commit**

```bash
git add web-pages/product-site/legacy.py web-pages/product-site/build.py web-pages/product-site/tests
git commit -S -m "feat(site): preserve legacy routes and unify navigation"
```

### Task 7: Complete Output Validation

**Files:**
- Create: `web-pages/product-site/validate.py`
- Modify: `web-pages/product-site/tests/test_output.py`
- Create: `.github/workflows/product-site.yml`

**Interfaces:**
- Consumes: a complete output directory
- Produces: `validate_output(output_dir: Path) -> list[str]`
- CLI exits 0 with `validated <page count> pages`; exits 1 and prints one stable error per line

- [ ] **Step 1: Write validation failure tests**

```python
def test_broken_internal_link_fails_validation(built_site):
    replace_href(built_site / 'index.html', '/deploy/', '/missing/')
    assert '/index.html: broken internal link /missing/' in validate_output(built_site)

def test_missing_language_peer_fails_validation(built_site):
    (built_site / 'en/deploy/vllm.html').unlink()
    assert any('missing hreflang peer' in error for error in validate_output(built_site))
```

- [ ] **Step 2: Verify tests fail**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_output.py -q`

Expected: FAIL because `validate_output` does not exist.

- [ ] **Step 3: Implement deterministic output checks**

Check internal links, duplicate ids, canonical/hreflang symmetry, sitemap equality, robots, JSON-LD parseability, asset hashes, image alt text, manifest routes, old indexed routes, deployment evidence markers, and the absence of external font/runtime dependencies.

- [ ] **Step 4: Add CI build job**

CI creates a Python 3.11 virtual environment, installs `requirements-site.txt`, runs all product-site tests, builds to a temporary directory, validates it, rebuilds, and compares recursive SHA-256 manifests to prove reproducibility.

- [ ] **Step 5: Run the complete static suite**

Run: `python3.11 -m pytest web-pages/product-site/tests -q && python3.11 web-pages/product-site/build.py --output /tmp/funasr-product-site && python3.11 web-pages/product-site/validate.py /tmp/funasr-product-site`

Expected: all tests pass and validator reports every generated and legacy page.

- [ ] **Step 6: Commit**

```bash
git add web-pages/product-site/validate.py web-pages/product-site/tests/test_output.py .github/workflows/product-site.yml
git commit -S -m "ci(site): validate reproducible product builds"
```

### Task 8: Atomic Release, Nginx Hardening, and Conversion Measurement

**Files:**
- Create: `web-pages/scripts/deploy-product-site.sh`
- Create: `web-pages/scripts/rollback-product-site.sh`
- Create: `web-pages/nginx/funasr.com.conf`
- Create: `web-pages/nginx/conversion-map.conf`
- Create: `web-pages/product-site/tests/test_release_scripts.py`

**Interfaces:**
- Deploy input: validated output directory and UTC release id `YYYYMMDDTHHMMSSZ`
- Deploy output: `/root/FunASR/web-pages/releases/<release id>/` and atomic `/root/FunASR/web-pages/current`
- Rollback input: an existing release id
- Fixed conversion routes: `/go/github`, `/go/docs`, `/go/releases`, `/go/deploy-vllm`, `/go/deploy-llama-cpp`

- [ ] **Step 1: Write shell-contract tests**

```python
def test_deploy_keeps_previous_release_and_switches_symlink(fake_server):
    deploy(fake_server, '20260726T120000Z')
    deploy(fake_server, '20260726T130000Z')
    assert release_ids(fake_server) == ['20260726T120000Z', '20260726T130000Z']
    assert current_release(fake_server) == '20260726T130000Z'

def test_conversion_map_has_no_open_redirects():
    routes = parse_conversion_map(CONVERSION_MAP)
    assert set(routes) == EXPECTED_FIXED_ROUTES
    assert all(target.startswith('https://github.com/') or target.startswith('https://funasr.com/') for target in routes.values())
```

- [ ] **Step 2: Verify tests fail**

Run: `python3.11 -m pytest web-pages/product-site/tests/test_release_scripts.py -q`

Expected: FAIL because scripts/config do not exist.

- [ ] **Step 3: Implement guarded release and rollback scripts**

The deploy script validates the output, copies it into a new release directory, verifies the copied manifest, backs up Nginx configuration with the release id, runs `nginx -t`, atomically changes `current`, reloads Nginx, and runs localhost HTTP checks. Any failure before symlink switch leaves the current release untouched; any failure after switch restores the captured previous target.

- [ ] **Step 4: Add hardened Nginx configuration**

Serve `current`, allow TLS 1.2/1.3 only, retain HSTS/frame/content-type/referrer headers, add a local-asset-compatible CSP and Permissions-Policy, disable cache for HTML, enable immutable cache for hashed assets, serve a bilingual 404, and write only fixed `/go/*` requests to `/var/log/nginx/funasr-conversions.log`.

- [ ] **Step 5: Test in a disposable Nginx prefix**

Run the configuration against a temporary Nginx prefix with a generated test certificate; assert `nginx -t`, canonical route status, cache headers, security headers, fixed redirect targets, conversion log entries, rollback, and retained releases.

- [ ] **Step 6: Commit**

```bash
git add web-pages/scripts web-pages/nginx web-pages/product-site/tests/test_release_scripts.py
git commit -S -m "ops(site): add atomic releases and hardened nginx"
```

### Task 9: Browser Verification and Production Rollout

**Files:**
- Create: `web-pages/product-site/tests/browser/product-site.spec.ts`
- Create: `web-pages/product-site/tests/browser/package.json`
- Create: `web-pages/product-site/tests/browser/playwright.config.ts`
- Create: `web-pages/product-site/tests/browser/screenshots/`
- Create: `docs/operations/funasr-com-site-release.md`

**Interfaces:**
- Consumes: locally served validated output and staged production release
- Produces: viewport screenshots, browser assertions, live smoke evidence, and rollback runbook

- [ ] **Step 1: Add browser assertions before deployment**

Tests cover 390x844, 768x1024, 1440x900, and 1920x1080. Assert no horizontal overflow, no intersecting visible controls, nonblank hero pixels, next-section visibility, keyboard menu operation, focus visibility, selector results, copy actions, language peers, old llama routes, reduced motion, and all fixed outbound redirects.

- [ ] **Step 2: Run local browser tests**

Run: `npx playwright test --config web-pages/product-site/tests/browser/playwright.config.ts`

Expected: PASS at all four viewports with committed reference screenshots reviewed for readable text and correct asset framing.

- [ ] **Step 3: Create an immutable production backup and staged release**

Archive the current live Nginx configuration and current static root with SHA-256 manifests under a timestamped backup. Upload the new build to a new release directory and run validator plus localhost checks before changing `current`.

- [ ] **Step 4: Switch production atomically**

Run `nginx -t`, switch `current`, reload Nginx, then verify public HTTPS for every manifest route, canonical/hreflang, assets, old routes, security/cache headers, sitemap, TLS 1.0/1.1 rejection, TLS 1.2/1.3 acceptance, and one fixed conversion event.

- [ ] **Step 5: Exercise rollback and restore the new release**

Switch to the captured previous release, verify home and blog routes, then switch back to the new release and repeat the public smoke test. Record both release ids and exact commands in `docs/operations/funasr-com-site-release.md`.

- [ ] **Step 6: Monitor first-day behavior**

Check Nginx 4xx/5xx rates, deployment page views, fixed GitHub/docs/release actions, and broken-link reports after 1 hour and 24 hours. Roll back immediately on elevated 5xx, missing indexed routes, unreadable mobile layout, or conversion redirect failure.

- [ ] **Step 7: Commit rollout evidence and open the FunASR PR**

```bash
git add web-pages/product-site/tests/browser docs/operations/funasr-com-site-release.md
git commit -S -m "test(site): verify product hub rollout"
git push origin codex/funasr-product-deployment-hub
gh pr create --repo modelscope/FunASR --base main --head codex/funasr-product-deployment-hub --title "feat(site): launch the FunASR deployment product hub" --body-file /tmp/funasr-product-site-pr.md
```

The PR body includes generated route count, exact test commands, screenshot links, production release id, rollback proof, claim/evidence policy, and follow-up metrics. The branch remains independently revertible by signed commits even though the site is already deployed from the exact reviewed build.

---

## Self-Review Results

- Spec coverage: registry/evidence, selector, bilingual routes, seven deployment pages, benchmarks, legacy preservation, navigation, measurement, atomic release, Nginx/TLS, browser checks, rollback, and first-day monitoring each map to an explicit task.
- Scope boundary: model/runtime implementation, public inference hosting, authentication, rate limiting, and Prometheus remain excluded.
- Type consistency: registry data flows through `load_registry`, `validate_registry`, `recommend`, `build`, `normalize_document`, and `validate_output` with stable names across tasks.
- Placeholder scan: the plan contains no deferred implementation markers; dynamic release ids are specified by an exact UTC format.
