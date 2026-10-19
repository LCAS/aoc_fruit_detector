#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
log_dir="$repo_root/scripts/data/training_output"
state_dir="${TMPDIR:-/tmp}/fruit-detector-tensorboard-${UID}"
mkdir -p "$log_dir" "$state_dir"
command -v tensorboard >/dev/null || {
    echo "TensorBoard is missing. Install it with: python3 -m pip install tensorboard" >&2
    exit 1
}
# Keep one background server per user across repeated startup commands.
nohup flock -n "$state_dir/server.lock" tensorboard \
    --logdir "$log_dir" --host 0.0.0.0 --port 6006 \
    >"$state_dir/server.log" 2>&1 </dev/null &
echo "TensorBoard: http://localhost:6006 (log: $state_dir/server.log)"
