# Audex

Simple personal project that extends https://github.com/ggml-org/whisper.cpp.git the main scope of this project is to generate text from .m4a files with a NN modell.
The code can handle different languages.


how to run:
./build/audex   --model models/ggml-large-v3.bin   --device cpu   --task transcribe   --language hu   --beam-size 8   input.m4a > output.txt
