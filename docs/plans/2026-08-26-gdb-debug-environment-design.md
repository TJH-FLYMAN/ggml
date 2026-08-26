# GDB Debug Environment Design

## Goal

Provide both command-line and VS Code entry points for debugging any executable under `build/bin` with GDB.

## Current State

- GDB 15.1 is installed at `/usr/bin/gdb`.
- The existing CMake build uses `CMAKE_BUILD_TYPE=Debug`.
- Executables under `build/bin` contain debug information and are not stripped.
- The existing VS Code launch configuration is fixed to `gpt-2-sched`.
- The existing VS Code build task compiles only the active source file and does not build the CMake project.

## Design

### Shared GDB settings

Add a repository-local GDB command file with settings useful for this C/C++ project, including pretty printing, readable array/string output, source-aware disassembly, and non-paginated output. Keeping the file inside the repository avoids changing the user's global `~/.gdbinit`.

### Command-line entry point

Add an executable `scripts/gdb-bin` wrapper. It will:

1. Resolve the repository root independently of the caller's current directory.
2. Accept an executable name from `build/bin`, with optional program arguments.
3. Reject missing, invalid, or non-executable targets with a useful message.
4. Start `/usr/bin/gdb` with the shared project settings and forward program arguments using `--args`.

For example:

```bash
scripts/gdb-bin gpt-2-sched
scripts/gdb-bin gpt-2-backend models/gpt-2-117M/ggml-model-gpt-2-117M.bin
```

### VS Code entry point

Replace the fixed launch entry with a generic `cppdbg` configuration. On launch, VS Code will prompt for the executable name beneath `build/bin`, use `/usr/bin/gdb`, set the repository root as the working directory, and apply equivalent GDB setup commands.

Program arguments remain an explicit JSON array in `launch.json`, because passing a prompted string would incorrectly collapse multiple arguments into one. The command-line wrapper remains the convenient entry point for arbitrary arguments.

### Build tasks

Replace the active-file compiler task with CMake Debug configure and incremental build tasks. The debug launch will run the incremental build task first so the executable and source information stay synchronized.

## Error Handling

The wrapper will print usage and list available executables when no program is specified. It will distinguish a missing target from a non-executable file and return a nonzero status without starting GDB.

## Verification

- Exercise wrapper validation for missing and unknown program names.
- Start `simple-ctx` through the wrapper in GDB batch mode, break at `main`, and confirm GDB resolves the source location.
- Parse the VS Code JSON-with-comments files after removing comments or inspect them with a JSONC-capable check.
- Run the configured CMake incremental build task command.
- Review the final diff and ensure the unrelated `get_gpt2.sh` modification remains untouched.
