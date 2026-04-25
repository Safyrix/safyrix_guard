#!/bin/sh
set -eu

KONG_ADMIN="${KONG_ADMIN:-http://kong-gateway:8001}"
SERVICE_URL="${SERVICE_URL:-http://analyzer:8000}"

echo "[init] waiting for Kong Admin at $KONG_ADMIN ..."
i=0
until curl -fsS "$KONG_ADMIN/status" >/dev/null 2>&1; do
  i=$((i+1))
  if [ "$i" -ge 60 ]; then
    echo "[init] Kong Admin not ready after 60 tries"
    exit 1
  fi
  sleep 1
done

echo "[init] upserting service analyzer -> $SERVICE_URL"
curl -fsS -X PUT "$KONG_ADMIN/services/analyzer" \
  --data "name=analyzer" \
  --data "url=$SERVICE_URL" >/dev/null

echo "[init] upserting route /analyzer -> analyzer"
curl -fsS -X PUT "$KONG_ADMIN/routes/analyzer-route" \
  --data "name=analyzer-route" \
  --data "paths[]=/analyzer" \
  --data "strip_path=true" \
  --data "service.name=analyzer" >/dev/null

echo "[init] done"
