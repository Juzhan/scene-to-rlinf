#!/usr/bin/env bash
# Record the commits of third_party/ checkouts (not tracked here),
# plus their local diffs, so a clean checkout can be rebuilt.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=docs/third_party
{
  echo "# path  remote  commit  dirty_files   (regenerate: bash $out/update.sh)"
  for d in third_party/*/; do
    d=${d%/}
    if git -C "$d" rev-parse --show-toplevel 2>/dev/null | grep -qx "$PWD/$d"; then
      echo "$d  $(git -C "$d" remote get-url origin 2>/dev/null || echo -)  $(git -C "$d" rev-parse HEAD)  $(git -C "$d" status --porcelain | wc -l)"
      name=$(basename "$d")
      if [ -n "$(git -C "$d" diff HEAD --stat)" ]; then git -C "$d" diff HEAD > "$out/$name.patch"; else rm -f "$out/$name.patch"; fi
    else
      echo "$d  (not a git checkout; copy separately)"
    fi
  done
} > "$out/LOCK.txt"
