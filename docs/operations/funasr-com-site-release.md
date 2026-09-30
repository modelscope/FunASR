# FunASR.com product-site release

## Production record

- Released: `2026-10-01` (Asia/Shanghai; verified at `2026-09-30T21:57:20Z`)
- Source commit: `460f00675499924fec0a14bdbeb7125ed505dbb2`
- Product release: `20260930T211757Z`
- Previous release: `20260918T065924Z`
- Current link: `/root/FunASR/web-pages/current`
- Release root: `/root/FunASR/web-pages/releases`
- Backup root: `/root/FunASR/web-pages/backups/product-site-20260930T211757Z`
- Build archive SHA-256: `408cc357883dfcae935faef87c95765cdfe249f5ff5916295c87d950b0f49221`
- Release manifest SHA-256: `d381634a0c25df10096cdd1befe2a03d4b4c3d968f8ff06ce7292a57acd56f8b`
- Active Nginx SHA-256: `4450f8616438fdf095728c12f50e0fa878a5afd6ccb63cd1cd4af4cc91f88233`

The source checkout used for the build matches the source commit's tree `73c82edb750f4e91253468779633fbf3200f7bfb`. The release ID was allocated during backup preparation, before the verified publication time above. This release publishes the merged speech-to-speech / SenseVoice source-install guide and bilingual ecosystem entries; it does not publish a new speech model or Python package.

## Backup evidence

The pre-release site and Nginx configuration were archived before the atomic switch.

| Artifact | SHA-256 |
| --- | --- |
| `live-dist.tar.gz` | `1df646e40da3ccdba3a254b6878aff40faf857be4d20d69c398bd07f1467d6ff` |
| `nginx.conf` | `4450f8616438fdf095728c12f50e0fa878a5afd6ccb63cd1cd4af4cc91f88233` |
| `build.tar.gz` | `408cc357883dfcae935faef87c95765cdfe249f5ff5916295c87d950b0f49221` |

The previous release remains intact at `20260918T065924Z`; its 206 pages passed the production validator before switching. The old release's file hashes were checked again immediately before deployment. No rollback was needed or exercised for this release. The existing Nginx master was gracefully reloaded, with its PID/start identity and configuration unchanged; no model service was restarted.

## Verification

- Exact-source GitHub Actions [Product site run `36747934507`](https://github.com/modelscope/FunASR/actions/runs/36747934507) passed all three jobs: legacy document links, build/validation, and browser checks. All five workflows on the release source commit completed successfully.
- Native pre-release checks: 521 product-site tests passed; selected root documentation tests had 100 passes and one collection skip because Sphinx was absent. The separate hosted legacy-document-link job passed.
- The 117-page product build plus preserved legacy routes passed validation for 206 total HTML pages on ind-gpu8, in production staging, and in the release script's copied staging tree. All 206 pre-existing HTML routes remain present.
- All 273 deployed files match the transferred per-file SHA-256 inventory. The archive is 8,470,216 bytes. Its output is byte-identical to the earlier browser-tested build.
- Thirteen public routes returned HTTP 200 with response bodies matching the release files: both homepages, ecosystem pages, community guides, deployment indexes, blog indexes and llama.cpp guides, plus `/donors.html`.
- A separate Chromium run from ind-gpu8 against the public domain passed four cases: Chinese/English at 1365px desktop and 390px mobile widths. It verified the integration card, localized guide link/anchor, pinned source recipe, no horizontal overflow and no JavaScript errors. Non-site requests were blocked; no audio/model conversation was run.
- All thirteen checked HTML responses are no-cache and include `X-Frame-Options`, `X-Content-Type-Options`, and HSTS headers.

The previous and new `deployment-manifest.json` files have the same checksum despite different HTML content. That manifest checksum alone does **not** prove publication freshness. Verify the actual [Chinese guide](https://www.funasr.com/docs/community.html#speech-to-speech), [English guide](https://www.funasr.com/en/docs/community.html#speech-to-speech), and per-file hashes when checking this release. Both guides must contain source revision `9e2ed1099190a4e4bc8a972b4a3488949ff1b9f6` and the explicit v1.0.0 exclusion.

## Release commands

The host runs an existing manually started Nginx master rather than the failed systemd unit. Discover its PID before each operation:

```bash
pgrep -a nginx
```

Deploy a new validated output directory:

```bash
PYTHONPATH=/root/.cache/funasr-ops/product-site-python \
NGINX_MASTER_PID=<master-pid> \
VALIDATOR=/root/FunASR/web-pages/ops/product-site/validate.py \
PYTHON_BIN=python3 \
/root/FunASR/web-pages/ops/product-site/deploy-product-site.sh \
  /path/to/validated-output YYYYMMDDTHHMMSSZ
```

Roll back to a product-site release:

```bash
PYTHONPATH=/root/.cache/funasr-ops/product-site-python \
NGINX_MASTER_PID=<master-pid> \
VALIDATOR=/root/FunASR/web-pages/ops/product-site/validate.py \
PYTHON_BIN=python3 \
/root/FunASR/web-pages/ops/product-site/rollback-product-site.sh YYYYMMDDTHHMMSSZ
```

Restore the pre-release Nginx configuration only if the new configuration is implicated:

```bash
cp -a /root/FunASR/web-pages/backups/product-site-20260930T211757Z/nginx.conf /etc/nginx/nginx.conf
nginx -t
kill -HUP <master-pid>
```

## Monitoring

Visible repository, documentation, and release links use the fixed `/go/github`,
`/go/docs`, and `/go/releases` routes. The JSON-LD `codeRepository` value remains
the direct GitHub URL so attribution does not change search metadata. Redirect
targets are defined only in `web-pages/nginx/conversion-map.conf`; never accept a
target from a query parameter.

Check after one hour and again after 24 hours:

```bash
tail -200 /var/log/nginx/error.log
tail -200 /var/log/nginx/funasr-conversions.log
curl -fsSI https://www.funasr.com/
curl -fsSI https://www.funasr.com/deploy/vllm.html
curl -fsSI https://www.funasr.com/blog/
```

Count non-smoke conversion requests by route:

```bash
grep -vE '"(FunASR release smoke|curl/)' /var/log/nginx/funasr-conversions.log \
  | awk '{print $7}' | sort | uniq -c | sort -nr
```

Roll back on elevated 5xx responses, missing indexed routes, mobile overflow, missing assets, invalid conversion redirects, or a failed static validation.
