mkdir -p build && cd build && rm -rf *
cmake .. && cmake --build .
cp progbox ..
cd ..