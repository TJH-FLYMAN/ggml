# gpt2 model size > 100M  , split to multiple files
cd models/gpt-2-117M

expected_md5="78ee53c64cc11b2ad87dce48d197689e"

if [ ! -f "ggml-model-gpt-2-117M.bin" ]; then
    cat ggml-model-gpt2-117M-* > model.bin
else
    md5_value=$(md5sum ggml-model-gpt-2-117M.bin | awk '{print $1}')
    if [ "$md5_value" = "$expected_md5" ]; then
        echo "ggml-model-gpt-2-117M.bin already exists!"
        echo "MD5  match"
        exit 0
    else
        echo "MD5 not match!"
        echo "Expected: $expected_md5"
        echo "Actual:   $md5_value"
        rm ggml-model-gpt-2-117M.bin
        cat ggml-model-gpt2-117M-* > model.bin
    fi
fi

md5_value=$(md5sum model.bin | awk '{print $1}')
echo "MD5: $md5_value"

if [ "$md5_value" = "$expected_md5" ]; then
    echo "MD5  match"
    mv model.bin ggml-model-gpt-2-117M.bin
else
    echo "MD5 not match!"
    echo "Expected: $expected_md5"
    echo "Actual:   $md5_value"
fi
