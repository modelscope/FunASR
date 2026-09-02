# FunASR.com product-site release

## Production record

- Released: `2026-09-02`
- Source commit: `e25fb88f4c06dfe186baee9d0d3f9a840346955d`
- Product release: `20260902T104600Z`
- Previous release: `20260901T065930Z`
- Current link: `/root/FunASR/web-pages/current`
- Release root: `/root/FunASR/web-pages/releases`
- Backup root: `/root/FunASR/web-pages/backups/product-site-20260902T103700Z`
- Build archive SHA-256: `4e224fbd23b55515bcbf7c5d02a7af1953ba9e02ea4221857c2bbf2f032e2d16`
- Release manifest SHA-256: `d73f247d951d8e49703d089ca7cca7de9077b73260f907f26c2f287b42e14276`
- Active Nginx SHA-256: `4450f8616438fdf095728c12f50e0fa878a5afd6ccb63cd1cd4af4cc91f88233`

The product-site tree was verified unchanged between the release source commit and the subsequent growth-snapshot documentation merge.

## Backup evidence

The pre-release site and Nginx configuration were archived before the atomic switch.

| Artifact | SHA-256 |
| --- | --- |
| `live-dist.tar.gz` | `4c8957d6f2212ea6753494078dcfbfe6abb5d24e1454587500e0af15c6758244` |
| `nginx.conf` | `4450f8616438fdf095728c12f50e0fa878a5afd6ccb63cd1cd4af4cc91f88233` |
| `build.tar.gz` | `4e224fbd23b55515bcbf7c5d02a7af1953ba9e02ea4221857c2bbf2f032e2d16` |

The previous release remains intact at `20260901T065930Z`. The release script's rollback trap was exercised twice during publish dry attempts, preserving that target before the final atomic switch.

## Verification

- Exact-main GitHub Actions run `33619555092` passed both product-site build/validation and Playwright browser jobs.
- Static validator: 110 pages passed on ind-gpu8 before transfer and again in the production staging directory.
- Public smoke: `/`, both ecosystem pages, `/deploy/`, `/blog/`, `/donors.html`, and both llama.cpp routes returned successfully.
- The Chinese and English ecosystem pages both contain the updated `37K+` social proof.
- Production HTML is no-cache and the checked response includes `X-Frame-Options`, `X-Content-Type-Options`, and HSTS headers.

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
cp -a /root/FunASR/web-pages/backups/product-site-20260726T171818Z/nginx.conf /etc/nginx/nginx.conf
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
