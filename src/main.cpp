#include "whisper.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace audex {
namespace {

int default_thread_count() {
  const unsigned int threads = std::thread::hardware_concurrency();
  return threads == 0 ? 4 : static_cast<int>(threads);
}

[[noreturn]] void print_usage_and_exit(const int exit_code) {
  std::ostream& stream = exit_code == 0 ? std::cout : std::cerr;

  stream
    << "Usage:\n"
    << "  audex --model <model-file> [options] <input-audio>\n\n"
    << "Options:\n"
    << "  --model <file>       Path to a whisper.cpp ggml model file\n"
    << "  --task <mode>        transcribe | translate (default: translate)\n"
    << "  --language <code>    Language code or name, or auto (default: auto)\n"
    << "  --device <name>      cpu | gpu (default: cpu)\n"
    << "  --gpu-device <id>    GPU device index for whisper.cpp (default: 0)\n"
    << "  --threads <n>        Number of inference threads (default: hardware concurrency)\n"
    << "  --beam-size <n>      Beam size, 1 = greedy decoding (default: 5)\n"
    << "  -h, --help           Show this help\n\n"
    << "Examples:\n"
    << "  audex --model models/ggml-large-v3.bin --task translate --language hu input.m4a\n"
    << "  audex --model models/ggml-large-v3.bin --task transcribe --language auto input.m4a\n";

  std::exit(exit_code);
}

std::string require_value(const int argc, char** argv, int& index, const std::string& option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument("missing value for " + option);
  }

  ++index;
  return argv[index];
}

Task parse_task(const std::string& value) {
  if (value == "transcribe") {
    return Task::kTranscribe;
  }
  if (value == "translate") {
    return Task::kTranslate;
  }

  throw std::invalid_argument("unsupported task '" + value +
                              "', expected 'transcribe' or 'translate'");
}

Device parse_device(const std::string& value) {
  if (value == "cpu") {
    return Device::kCpu;
  }
  if (value == "gpu") {
    return Device::kGpu;
  }

  throw std::invalid_argument("unsupported device '" + value + "', expected 'cpu' or 'gpu'");
}

WhisperRunOptions parse_arguments(const int argc, char** argv) {
  if (argc <= 1) {
    print_usage_and_exit(1);
  }

  WhisperRunOptions options;
  options.threads = default_thread_count();

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];

    if (argument == "-h" || argument == "--help") {
      print_usage_and_exit(0);
    }

    if (argument == "--model") {
      options.model_path = require_value(argc, argv, index, argument);
      continue;
    }

    if (argument == "--task") {
      options.task = parse_task(require_value(argc, argv, index, argument));
      continue;
    }

    if (argument == "--language") {
      options.language = require_value(argc, argv, index, argument);
      continue;
    }

    if (argument == "--device") {
      options.device = parse_device(require_value(argc, argv, index, argument));
      continue;
    }

    if (argument == "--gpu-device") {
      options.gpu_device = std::stoi(require_value(argc, argv, index, argument));
      if (options.gpu_device < 0) {
        throw std::invalid_argument("--gpu-device must be zero or greater");
      }
      continue;
    }

    if (argument == "--threads") {
      options.threads = std::stoi(require_value(argc, argv, index, argument));
      if (options.threads <= 0) {
        throw std::invalid_argument("--threads must be greater than zero");
      }
      continue;
    }

    if (argument == "--beam-size") {
      const int beam_size = std::stoi(require_value(argc, argv, index, argument));
      if (beam_size <= 0) {
        throw std::invalid_argument("--beam-size must be greater than zero");
      }
      options.beam_size = static_cast<std::size_t>(beam_size);
      continue;
    }

    if (!argument.empty() && argument.front() == '-') {
      throw std::invalid_argument("unknown option: " + argument);
    }

    if (!options.input_path.empty()) {
      throw std::invalid_argument("only one input audio file can be provided");
    }

    options.input_path = argument;
  }

  if (options.model_path.empty()) {
    throw std::invalid_argument("--model is required");
  }

  if (options.input_path.empty()) {
    throw std::invalid_argument("input audio path is required");
  }

  return options;
}

}  // namespace
}  // namespace audex

int main(int argc, char** argv) {
  try {
    const audex::WhisperRunOptions options = audex::parse_arguments(argc, argv);

    if (options.task == audex::Task::kTranslate) {
      std::cerr << "task=translate uses Whisper's English-only translation mode\n";
    }

    const audex::WhisperRunResult result = audex::run_whisper_file(options);
    std::cout << result.text << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
