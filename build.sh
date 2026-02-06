pushd .
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
popd

# if [ ! -f "models/gpt-2-117M/ggml-model-gpt-2-117M.bin" ]; then
#     ./get_gpt2.sh
#     cp -r models build/models
# fi
