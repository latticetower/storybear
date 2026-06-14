#!/usr/bin/env bash
# Cold-start all three Storybear model services on Modal and wait until each is
# ready to serve. Modal boots containers lazily, so this pokes each service's
# readiness endpoint and retries until it returns HTTP 200 (or a deadline hits).
# The three services are started in parallel.
#
# Usage:
#   ./modal/coldstart.sh <workspace>
#   MODAL_WORKSPACE=<workspace> ./modal/coldstart.sh
#
# Or override individual URLs (skips workspace-based construction). These also
# fall back to the STORYBEAR_* vars the pipeline already uses:
#   VLLM_URL=...  LLAMACPP_URL=...  FLUX_URL=...  ./modal/coldstart.sh
#
# The endpoints require Modal proxy auth, so provide a proxy auth token. These
# fall back to the STORYBEAR_* vars the pipeline uses:
#   MODAL_KEY=wk-...  MODAL_SECRET=ws-...  ./modal/coldstart.sh
#
# Tunables (env):
#   DEADLINE      overall wait per service, seconds (default 900)
#   INTERVAL      seconds between polls (default 5)
#   REQ_TIMEOUT   per-request timeout, seconds (default 30)

set -uo pipefail

WORKSPACE="${1:-${MODAL_WORKSPACE:-}}"

DEADLINE="${DEADLINE:-900}"
INTERVAL="${INTERVAL:-5}"
REQ_TIMEOUT="${REQ_TIMEOUT:-30}"

# Resolve URLs: explicit override > STORYBEAR_* fallback > built from workspace.
VLLM_URL="${VLLM_URL:-${STORYBEAR_VLM_URL:-}}"
LLAMACPP_URL="${LLAMACPP_URL:-}"
FLUX_URL="${FLUX_URL:-${STORYBEAR_FLUX_URL:-}}"

# Proxy auth token: explicit override > STORYBEAR_* fallback. Required because
# the endpoints enforce proxy auth; without it every probe gets HTTP 401.
MODAL_KEY="${MODAL_KEY:-${STORYBEAR_MODAL_KEY:-}}"
MODAL_SECRET="${MODAL_SECRET:-${STORYBEAR_MODAL_SECRET:-}}"

# Build curl header args once; both id and secret must be present to authenticate.
AUTH_ARGS=()
if [[ -n "$MODAL_KEY" && -n "$MODAL_SECRET" ]]; then
  AUTH_ARGS=(-H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET")
else
  echo "WARNING: MODAL_KEY/MODAL_SECRET not set; probes will get HTTP 401 from the" >&2
  echo "         auth-protected endpoints. Set them (or STORYBEAR_MODAL_KEY/SECRET)." >&2
fi

build_url() { printf 'https://%s--%s.modal.run' "$WORKSPACE" "$1"; }

if [[ -z "$VLLM_URL" || -z "$LLAMACPP_URL" || -z "$FLUX_URL" ]]; then
  if [[ -z "$WORKSPACE" ]]; then
    echo "Provide a workspace (arg or MODAL_WORKSPACE), or set VLLM_URL / LLAMACPP_URL / FLUX_URL." >&2
    exit 2
  fi
  VLLM_URL="${VLLM_URL:-$(build_url storybear-minicpm-v-serve)}"
  LLAMACPP_URL="${LLAMACPP_URL:-$(build_url storybear-minicpm-v-llamacpp-serve)}"
  # NOTE: class-based asgi_app endpoints include the class name in the subdomain
  # (FluxKlein.web -> ...-fluxklein-web), unlike the function-based VLM servers.
  FLUX_URL="${FLUX_URL:-$(build_url storybear-flux-klein-fluxklein-web)}"
fi

# Normalize: the readiness probes append their own paths.
VLLM_URL="${VLLM_URL%/}";       VLLM_URL="${VLLM_URL%/v1}"
LLAMACPP_URL="${LLAMACPP_URL%/}"; LLAMACPP_URL="${LLAMACPP_URL%/v1}"
FLUX_URL="${FLUX_URL%/}";        FLUX_URL="${FLUX_URL%/edit}"

# Poll one service until its probe returns HTTP 200.
# Args: <name> <probe-url>
wait_for() {
  local name="$1" probe="$2"
  local start code now
  start=$(date +%s)
  echo "[$name] cold starting -> $probe"
  while :; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time "$REQ_TIMEOUT" \
      "${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"}" "$probe")
    if [[ "$code" == "200" ]]; then
      now=$(date +%s)
      echo "[$name] READY in $((now - start))s"
      return 0
    fi
    if [[ "$code" == "401" || "$code" == "403" ]]; then
      echo "[$name] ERROR HTTP $code at $probe — missing or invalid proxy auth token." >&2
      echo "[$name] Set MODAL_KEY/MODAL_SECRET (or STORYBEAR_MODAL_KEY/SECRET)." >&2
      return 1
    fi
    if [[ "$code" == "404" ]]; then
      echo "[$name] ERROR HTTP 404 at $probe — likely 'modal-http: invalid function call'." >&2
      echo "[$name] Check the URL and that the app is deployed (modal app list)." >&2
      return 1
    fi
    now=$(date +%s)
    if (( now - start >= DEADLINE )); then
      echo "[$name] TIMEOUT after ${DEADLINE}s (last HTTP $code)." >&2
      return 1
    fi
    printf '[%s] not ready (HTTP %s), %ss elapsed...\n' "$name" "$code" "$((now - start))"
    sleep "$INTERVAL"
  done
}

echo "Cold-starting services:"
echo "  vllm     = $VLLM_URL"
echo "  llamacpp = $LLAMACPP_URL"
echo "  flux     = $FLUX_URL"
echo

# Both VLMs expose /health, which returns 200 only once the model is fully
# loaded (/v1/models can answer earlier, while the model is still loading).
# FLUX is a FastAPI app, probed via /docs.
wait_for "vllm"     "$VLLM_URL/health"     & pid_vllm=$!
wait_for "llamacpp" "$LLAMACPP_URL/health" & pid_llamacpp=$!
wait_for "flux"     "$FLUX_URL/docs"       & pid_flux=$!

rc=0
wait "$pid_vllm"     || rc=1
wait "$pid_llamacpp" || rc=1
wait "$pid_flux"     || rc=1

echo
if (( rc == 0 )); then
  echo "All services ready."
else
  echo "One or more services failed to become ready." >&2
fi
exit "$rc"
