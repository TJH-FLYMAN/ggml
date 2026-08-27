#!/usr/bin/env bash

# GPT-2 model files larger than 100 MB are split into multiple files.

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

merge_and_verify_model() {
    model_dir=$1
    output_file=$2
    part_prefix=$3
    expected_md5=$4
    temporary_file="${output_file}.tmp"

    (
        cd "$model_dir" || exit 1

        if [ -f "$output_file" ]; then
            md5_value=$(md5sum "$output_file" | awk '{print $1}')
            if [ "$md5_value" = "$expected_md5" ]; then
                echo "$output_file already exists!"
                echo "MD5 match: $md5_value"
                exit 0
            fi

            echo "$output_file MD5 not match; rebuilding from split files."
            echo "Expected: $expected_md5"
            echo "Actual:   $md5_value"
        fi

        if ! cat "${part_prefix}"-* > "$temporary_file"; then
            echo "Failed to concatenate ${part_prefix}-*" >&2
            rm -f "$temporary_file"
            exit 1
        fi

        md5_value=$(md5sum "$temporary_file" | awk '{print $1}')
        echo "$output_file MD5: $md5_value"

        if [ "$md5_value" = "$expected_md5" ]; then
            echo "MD5 match"
            mv "$temporary_file" "$output_file"
        else
            echo "MD5 not match!"
            echo "Expected: $expected_md5"
            echo "Actual:   $md5_value"
            rm -f "$temporary_file"
            exit 1
        fi
    )
}

merge_and_verify_model \
    "$script_dir/models/gpt-2-117M" \
    "ggml-model-gpt-2-117M.bin" \
    "ggml-model-gpt2-117M" \
    "78ee53c64cc11b2ad87dce48d197689e" || exit 1

merge_and_verify_model \
    "$script_dir/models/gpt2-Q4KM" \
    "gpt2.Q4_K_M.gguf" \
    "gpt2.Q4_K_M.gguf" \
    "50e5b09dd50b15483ebef946d2c13bb6" || exit 1
