# Audex

Simple personal project that extends [Whisper.cpp](https://github.com/ggml-org/whisper.cpp.git) the main scope of this project is to generate text from .m4a files with a NN modell.
The code can handle different languages.

## build

after cloning

Cuda:
cmake -S . -B build -DGGML_CUDA=1

Vulkan:
cmake -S . -B build -DGGML_VULKAN=1

cmake --build build -j

how to run:
./build/audex   --model models/ggml-large-v3.bin   --device cpu   --task transcribe   --language hu   --beam-size 8   input.m4a > output.txt
