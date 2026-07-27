# FunASR.com product-site release

## Production record

- Released: `2026-07-26`
- Source commit: `5a51d04d684528f1884f7c0a6e95d33748bb3805`
- Product release: `20260726T171818Z`
- Previous release: `20260726T171800Z`
- Current link: `/root/FunASR/web-pages/current`
- Release root: `/root/FunASR/web-pages/releases`
- Backup root: `/root/FunASR/web-pages/backups/product-site-20260726T171818Z`
- Build archive SHA-256: `f1184ff9c1b8b7130d146549ed911e2349eee93f6c0d745d526da02cb5412280`
- Active Nginx SHA-256: `241eebf7f0b2aa7a148ccb2c8c8ab58818efba9a37b8dc7392b25076d18a994d`

The build from the final source commit was byte-identical to the reviewed build from `caa948d9`. The final two commits only change release-script behavior.

## Backup evidence

The pre-release site and Nginx configuration were archived before the first switch.

| Artifact | SHA-256 |
| --- | --- |
| `live-dist.tar.gz` | `61f79a97d7a6677b1b11e01b18f03cdff1bef617960769549256244e489687f5` |
| `nginx.conf` | `46da00bd6c51b19b22a21658f141417653327fb670bf58cb574311c337e469dc` |
| `live-dist.sha256` | `a9e26f1b50deb759902a011351aefc70e356cdccdca3f8ce571b88ffeddb7395` |

The previous release was checked against the backup manifest before the rollback exercise: 289 immutable files matched. `stats/data.json` is excluded from that comparison because the legacy statistics process writes it continuously.

## Verification

- Python suite: 41 passed.
- Playwright suite: 7 passed at 390x844, 768x1024, 1440x900, and 1920x1080.
- Static validator: 92 HTML pages passed before upload, after upload, and after release copy.
- Public smoke: 25 pages, 5 fixed redirects, security headers, HTML no-cache, and one hashed immutable asset passed.
- TLS: TLS 1.2 and TLS 1.3 handshakes passed; the 443 server only enables those protocols.
- Production screenshots at 390x844 and 1440x900 were byte-identical to the committed reference screenshots.
- Rollback was exercised from `20260726T171818Z` to `20260726T171800Z`, the legacy home/blog/donor/llama routes passed, and the product release was restored and revalidated.

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
