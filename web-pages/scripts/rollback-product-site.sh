#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 YYYYMMDDTHHMMSSZ" >&2
  exit 2
fi

release_id=$1
if [[ ! $release_id =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "invalid release id: $release_id" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
site_base=${SITE_BASE:-/root/FunASR/web-pages}
current_link=$site_base/current
target=$site_base/releases/$release_id
validator=${VALIDATOR:-$script_dir/../product-site/validate.py}
python_bin=${PYTHON_BIN:-python3}
nginx_bin=${NGINX_BIN:-nginx}
nginx_master_pid=${NGINX_MASTER_PID:-}
curl_bin=${CURL_BIN:-curl}
smoke_base_url=${SMOKE_BASE_URL:-https://www.funasr.com}
smoke_routes=${SMOKE_ROUTES:-/ /deploy/ /blog/ /donors.html}
temporary_link=$site_base/.current.rollback.$$
lock_dir=$site_base/.product-site-release.lock
previous_target=
switched=0

reload_nginx() {
  if [[ -n $nginx_master_pid ]]; then
    kill -HUP "$nginx_master_pid"
  else
    "$nginx_bin" -s reload
  fi
}

cleanup() {
  rm -f -- "$temporary_link"
  rmdir -- "$lock_dir" 2>/dev/null || true
}

on_error() {
  status=$?
  set +e
  if [[ $switched -eq 1 && -n $previous_target ]]; then
    ln -s -- "$previous_target" "$temporary_link"
    mv -Tf -- "$temporary_link" "$current_link"
    reload_nginx >/dev/null 2>&1 || true
  fi
  cleanup
  echo "rollback failed; previous site restored" >&2
  exit "$status"
}

trap on_error ERR INT TERM
trap cleanup EXIT

if [[ ! -d $target ]]; then
  echo "release does not exist: $release_id" >&2
  exit 1
fi
if ! mkdir -- "$lock_dir" 2>/dev/null; then
  echo "another product-site release is active" >&2
  exit 1
fi

"$python_bin" "$validator" "$target"
"$nginx_bin" -t
if [[ -L $current_link ]]; then
  previous_target=$(readlink -- "$current_link")
fi
ln -s -- "$target" "$temporary_link"
mv -Tf -- "$temporary_link" "$current_link"
switched=1
reload_nginx

for route in $smoke_routes; do
  "$curl_bin" --fail --silent --show-error --max-time 15 "$smoke_base_url$route" >/dev/null
done

switched=0
echo "rolled back to $release_id"
