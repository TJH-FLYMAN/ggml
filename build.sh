pushd .
# # rm -rf build
# mkdir build && cd build
cd build
cmake --build . --config Release -j 8
popd

