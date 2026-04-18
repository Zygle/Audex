#pragma once

#include <cstddef>
#include <string>

namespace audex {

enum class Task {
  kTranscribe,
  kTranslate,
};

enum class Device {
  kCpu,
  kGpu,
};

struct WhisperRunOptions {
  std::string input_path;
  std::string model_path;
  std::string language = "auto";
  int threads = 4;
  int gpu_device = 0;
  std::size_t beam_size = 5;
  Task task = Task::kTranslate;
  Device device = Device::kCpu;
};

struct WhisperRunResult {
  std::string detected_language;
  std::string text;
};

WhisperRunResult run_whisper_file(const WhisperRunOptions& options);
std::string task_to_string(Task task);

}  // namespace audex
