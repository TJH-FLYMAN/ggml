# GDB Debug Environment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Provide command-line and VS Code entry points that can build and debug any executable under `build/bin` with GDB.

**Architecture:** Store common debugger behavior in a repository-local GDB command file. A small Bash wrapper validates and launches command-line sessions, while VS Code uses a generic prompted executable name and CMake Debug tasks.

**Tech Stack:** Bash, GDB 15, VS Code `cppdbg`, CMake, JSON, `jq`

---

### Task 1: Command-line GDB wrapper

**Files:**
- Create: `tests/test-gdb-bin.sh`
- Create: `gdb/ggml.gdb`
- Create: `scripts/gdb-bin`

**Step 1: Write the failing wrapper test**

Create `tests/test-gdb-bin.sh`:

```bash
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
    output="$($@ 2>&1)"
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
```

**Step 2: Run the test to verify it fails**

Run:

```bash
bash tests/test-gdb-bin.sh
```

Expected: FAIL because `scripts/gdb-bin` does not exist.

**Step 3: Add shared GDB settings**

Create `gdb/ggml.gdb`:

```gdb
set pagination off
set breakpoint pending on
set print pretty on
set print object on
set print elements 200
set print characters 200
set print demangle on
set disassembly-flavor intel
set disassemble-next-line auto
set listsize 20
```

**Step 4: Implement the minimal wrapper**

Create `scripts/gdb-bin`:

```bash
#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
bin_dir="$repo_root/build/bin"
gdb_commands="$repo_root/gdb/ggml.gdb"

usage() {
    printf 'Usage: %s <program-name> [program-arguments...]\n' "${0##*/}"
}

list_programs() {
    printf 'Available programs:\n'
    if [[ -d "$bin_dir" ]]; then
        find "$bin_dir" -maxdepth 1 -type f -perm -111 -printf '  %f\n' | sort
    fi
}

if [[ $# -eq 0 ]]; then
    usage >&2
    list_programs >&2
    exit 2
fi

case "$1" in
    -h|--help)
        usage
        list_programs
        exit 0
        ;;
    */*)
        printf 'error: program must be a name from build/bin, not a path: %s\n' "$1" >&2
        exit 2
        ;;
esac

program="$1"
shift
target="$bin_dir/$program"

if [[ ! -e "$target" ]]; then
    printf 'error: program %s not found under %s\n' "$program" "$bin_dir" >&2
    list_programs >&2
    exit 2
fi

if [[ ! -f "$target" || ! -x "$target" ]]; then
    printf 'error: target is not an executable file: %s\n' "$target" >&2
    exit 2
fi

gdb_bin="${GDB:-/usr/bin/gdb}"
if [[ ! -x "$gdb_bin" ]]; then
    printf 'error: GDB is not executable: %s\n' "$gdb_bin" >&2
    exit 2
fi

cd "$repo_root"
exec "$gdb_bin" -x "$gdb_commands" --args "$target" "$@"
```

Make both shell files executable:

```bash
chmod +x scripts/gdb-bin tests/test-gdb-bin.sh
```

**Step 5: Run wrapper tests**

Run:

```bash
bash -n scripts/gdb-bin tests/test-gdb-bin.sh
bash tests/test-gdb-bin.sh
```

Expected: syntax check exits 0 and the test prints `PASS: scripts/gdb-bin validation and argument forwarding`.

**Step 6: Commit the command-line entry point**

```bash
git add gdb/ggml.gdb scripts/gdb-bin tests/test-gdb-bin.sh
git commit -m "feat: add project GDB command-line launcher"
```

### Task 2: Generic VS Code GDB launch and CMake tasks

**Files:**
- Modify: `.vscode/launch.json`
- Modify: `.vscode/tasks.json`

**Step 1: Write failing structural assertions**

Run the assertions before changing the files:

```bash
jq -e '.configurations[0].program == "${workspaceFolder}/build/bin/${input:programName}"' .vscode/launch.json
jq -e '.inputs[] | select(.id == "programName")' .vscode/launch.json
jq -e '.tasks[] | select(.label == "CMake: build Debug" and .dependsOn == "CMake: configure Debug")' .vscode/tasks.json
```

Expected: at least one command fails because the current launch is fixed to `gpt-2-sched`, there is no input prompt, and there are no CMake tasks.

**Step 2: Replace the fixed launch configuration**

Write `.vscode/launch.json` as strict JSON:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "GDB: Debug build/bin program",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/bin/${input:programName}",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/bin/gdb",
            "preLaunchTask": "CMake: build Debug",
            "setupCommands": [
                {
                    "description": "Load repository GDB settings",
                    "text": "-interpreter-exec console \"source ${workspaceFolder}/gdb/ggml.gdb\"",
                    "ignoreFailures": false
                },
                {
                    "description": "Enable GDB pretty printing",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        }
    ],
    "inputs": [
        {
            "id": "programName",
            "type": "promptString",
            "description": "Executable name under build/bin",
            "default": "simple-ctx"
        }
    ]
}
```

**Step 3: Replace the active-file task with CMake Debug tasks**

Write `.vscode/tasks.json` as strict JSON:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "CMake: configure Debug",
            "type": "process",
            "command": "/usr/bin/cmake",
            "args": [
                "-S",
                "${workspaceFolder}",
                "-B",
                "${workspaceFolder}/build",
                "-DCMAKE_BUILD_TYPE=Debug"
            ],
            "problemMatcher": []
        },
        {
            "label": "CMake: build Debug",
            "type": "process",
            "command": "/usr/bin/cmake",
            "args": [
                "--build",
                "${workspaceFolder}/build",
                "--parallel"
            ],
            "dependsOn": "CMake: configure Debug",
            "dependsOrder": "sequence",
            "problemMatcher": [
                "$gcc"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

**Step 4: Validate JSON and required links**

Run:

```bash
jq empty .vscode/launch.json .vscode/tasks.json
jq -e '.configurations[0].program == "${workspaceFolder}/build/bin/${input:programName}"' .vscode/launch.json
jq -e '.configurations[0].preLaunchTask == "CMake: build Debug"' .vscode/launch.json
jq -e '.inputs[] | select(.id == "programName")' .vscode/launch.json
jq -e '.tasks[] | select(.label == "CMake: build Debug" and .dependsOn == "CMake: configure Debug")' .vscode/tasks.json
```

Expected: all commands exit 0.

**Step 5: Commit the VS Code configuration**

```bash
git add .vscode/launch.json .vscode/tasks.json
git commit -m "feat: configure generic VS Code GDB debugging"
```

### Task 3: End-to-end verification

**Files:**
- Verify only; no file changes expected

**Step 1: Run an incremental Debug build**

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

Expected: configuration and build finish successfully.

**Step 2: Verify GDB reaches `main` with source information**

Run:

```bash
scripts/gdb-bin simple-ctx <<'GDB'
set confirm off
break main
run
bt 1
quit
GDB
```

Expected: GDB reports a breakpoint in `main` in `examples/simple/simple-ctx.cpp` and prints a one-frame backtrace with a source line.

**Step 3: Run all static and wrapper checks**

Run:

```bash
bash -n scripts/gdb-bin tests/test-gdb-bin.sh
bash tests/test-gdb-bin.sh
jq empty .vscode/launch.json .vscode/tasks.json
git diff --check HEAD~2..HEAD
```

Expected: all checks exit 0.

**Step 4: Review scope and user changes**

Run:

```bash
git status --short
git log -4 --oneline
```

Expected: only the user's pre-existing `get_gpt2.sh` modification remains uncommitted, and recent commits contain the design, wrapper, and VS Code configuration.
