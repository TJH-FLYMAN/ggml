#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
wrapper="$repo_root/scripts/gdb-bin"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

assert_contains() {
    [[ "$output" == *"$1"* ]] || fail "expected output to contain: $1"
}

run_wrapper() {
    set +e
    output="$("$@" 2>&1)"
    status=$?
    set -e
}

run_wrapper "$wrapper"
[[ $status -eq 2 ]] || fail "missing program should exit 2, got $status"
assert_contains "Usage:"
assert_contains "Available programs:"

run_wrapper "$wrapper" does-not-exist
[[ $status -eq 2 ]] || fail "unknown program should exit 2, got $status"
assert_contains "not found under"

run_wrapper "$wrapper" ../simple-ctx
[[ $status -eq 2 ]] || fail "path input should exit 2, got $status"
assert_contains "must be a name"

run_wrapper "$wrapper" --help
[[ $status -eq 0 ]] || fail "--help should exit 0, got $status"
assert_contains "Usage:"

run_wrapper env GDB=/bin/echo "$wrapper" simple-ctx --flag "two words"
[[ $status -eq 0 ]] || fail "valid program should reach GDB, got $status"
assert_contains "-x $repo_root/gdb/ggml.gdb"
assert_contains "--args $repo_root/build/bin/simple-ctx --flag two words"

printf 'PASS: scripts/gdb-bin validation and argument forwarding\n'
