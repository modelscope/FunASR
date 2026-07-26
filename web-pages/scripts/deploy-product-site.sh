#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 OUTPUT_DIR YYYYMMDDTHHMMSSZ" >&2
  exit 2
fi

output_dir=$1
release_id=$2
if [[ ! $release_id =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "invalid release id: $release_id" >&2
  exit 2
fi
if [[ ! -d $output_dir ]]; then
  echo "output directory does not exist: $output_dir" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
site_base=${SITE_BASE:-/root/FunASR/web-pages}
releases_dir=$site_base/releases
current_link=$site_base/current
validator=${VALIDATOR:-$script_dir/../product-site/validate.py}
python_bin=${PYTHON_BIN:-python3}
nginx_config=${NGINX_CONFIG:-/etc/nginx/nginx.conf}
nginx_bin=${NGINX_BIN:-nginx}
nginx_master_pid=${NGINX_MASTER_PID:-}
curl_bin=${CURL_BIN:-curl}
smoke_base_url=${SMOKE_BASE_URL:-https://www.funasr.com}
smoke_routes=${SMOKE_ROUTES:-/ /deploy/ /blog/ /donors.html}
destination=$releases_dir/$release_id
staging=$releases_dir/.$release_id.staging.$$
temporary_link=$site_base/.current.$release_id.$$
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
  rm -rf -- "$staging"
  rm -f -- "$temporary_link"
  rmdir -- "$lock_dir" 2>/dev/null || true
}

restore_previous() {
  if [[ $switched -ne 1 ]]; then
    return
  fi
  if [[ -n $previous_target ]]; then
    ln -s -- "$previous_target" "$temporary_link"
    mv -Tf -- "$temporary_link" "$current_link"
  else
    rm -f -- "$current_link"
  fi
  reload_nginx >/dev/null 2>&1 || true
}

on_error() {
  status=$?
  set +e
  restore_previous
  cleanup
  echo "release failed; previous site restored" >&2
  exit "$status"
}

trap on_error ERR INT TERM
trap cleanup EXIT

mkdir -p -- "$site_base"
if ! mkdir -- "$lock_dir" 2>/dev/null; then
  echo "another product-site release is active" >&2
  exit 1
fi
if [[ -e $destination ]]; then
  echo "release already exists: $release_id" >&2
  exit 1
fi

"$python_bin" "$validator" "$output_dir"
mkdir -p -- "$releases_dir" "$site_base/nginx-backups" "$staging"
cp -a -- "$output_dir/." "$staging/"
"$python_bin" "$validator" "$staging"
cmp --silent "$output_dir/deployment-manifest.json" "$staging/deployment-manifest.json"
mv -- "$staging" "$destination"
cp -a -- "$nginx_config" "$site_base/nginx-backups/nginx.conf.$release_id"

"$nginx_bin" -t
if [[ -L $current_link ]]; then
  previous_target=$(readlink -- "$current_link")
fi
ln -s -- "$destination" "$temporary_link"
mv -Tf -- "$temporary_link" "$current_link"
switched=1
reload_nginx

for route in $smoke_routes; do
  "$curl_bin" --fail --silent --show-error --max-time 15 "$smoke_base_url$route" >/dev/null
done

switched=0
echo "released $release_id"
