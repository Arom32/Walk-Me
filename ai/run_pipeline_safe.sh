#!/usr/bin/env bash
# pipeline.py를 cgroup 메모리 상한(스왑 금지) 안에서 실행한다.
#
# 한도를 넘으면 스왑 스래싱으로 컴퓨터 전체가 느려지는 대신, 이 프로세스가
# 즉시 OOM-kill되고 종료한다. 기본 16G는 WALKME_PIPELINE_MEM_LIMIT로 조절.
# walkme-llm conda env가 이미 activate된 상태에서 실행할 것.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIMIT="${WALKME_PIPELINE_MEM_LIMIT:-16G}"

if command -v systemd-run >/dev/null 2>&1; then
  exec systemd-run --user --scope --collect \
    -p MemoryMax="$LIMIT" -p MemorySwapMax=0 -- \
    python "$SCRIPT_DIR/pipeline.py" "$@"
else
  echo "[경고] systemd-run 없음 — 메모리 한도 없이 실행합니다." >&2
  exec python "$SCRIPT_DIR/pipeline.py" "$@"
fi
