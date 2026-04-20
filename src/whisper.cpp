#include "whisper.hpp"

#include "audio.hpp"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <whisper.h>

namespace audex {
namespace {

using WhisperContextPtr = std::unique_ptr<whisper_context, decltype(&whisper_free)>;

class TerminalProgressBar {
 public:
  explicit TerminalProgressBar(const bool enabled) : enabled_(enabled) {}

  void update(const int progress) {
    if (!enabled_) {
      return;
    }

    const int clamped_progress = std::max(0, std::min(100, progress));

    std::lock_guard<std::mutex> lock(mutex_);
    if (finished_ || clamped_progress == last_progress_) {
      return;
    }

    last_progress_ = clamped_progress;

    constexpr int kBarWidth = 40;
    const int filled = (clamped_progress * kBarWidth) / 100;

    std::cerr << "\r[";
    for (int index = 0; index < kBarWidth; ++index) {
      std::cerr << (index < filled ? '#' : '-');
    }
    std::cerr << "] " << std::setw(3) << clamped_progress << '%' << std::flush;
  }

  void finish() {
    if (!enabled_) {
      return;
    }

    update(100);

    std::lock_guard<std::mutex> lock(mutex_);
    if (!finished_) {
      finished_ = true;
      std::cerr << '\n';
    }
  }

 private:
  bool enabled_ = true;
  bool finished_ = false;
  int last_progress_ = -1;
  std::mutex mutex_;
};

std::string trim(const std::string& value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
    return std::isspace(c) != 0;
  });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
    return std::isspace(c) != 0;
  }).base();

  if (first >= last) {
    return {};
  }

  return std::string(first, last);
}

bool language_is_auto(const std::string& language) {
  return language.empty() || language == "auto";
}

void whisper_progress_callback(whisper_context*,
                               whisper_state*,
                               const int progress,
                               void* user_data) {
  auto* progress_bar = static_cast<TerminalProgressBar*>(user_data);
  if (progress_bar != nullptr) {
    progress_bar->update(progress);
  }
}

void append_segment_text(std::string& full_text, const std::string& segment_text) {
  const std::string cleaned_segment = trim(segment_text);
  if (cleaned_segment.empty()) {
    return;
  }

  if (full_text.empty()) {
    full_text = cleaned_segment;
    return;
  }

  const unsigned char first = static_cast<unsigned char>(cleaned_segment.front());
  const bool starts_with_punctuation =
    std::ispunct(first) != 0 && cleaned_segment.front() != '"' && cleaned_segment.front() != '\'';

  if (!starts_with_punctuation && !std::isspace(static_cast<unsigned char>(full_text.back()))) {
    full_text.push_back(' ');
  }

  full_text += cleaned_segment;
}

WhisperContextPtr load_context(const WhisperRunOptions& options) {
  whisper_context_params context_params = whisper_context_default_params();
  context_params.use_gpu = options.device == Device::kGpu;
  context_params.gpu_device = options.gpu_device;

  whisper_context* context =
    whisper_init_from_file_with_params(options.model_path.c_str(), context_params);
  if (context == nullptr) {
    throw std::runtime_error("failed to load whisper.cpp model from '" + options.model_path + "'");
  }

  return WhisperContextPtr(context, whisper_free);
}

whisper_full_params build_full_params(whisper_context* context,
                                      const WhisperRunOptions& options,
                                      TerminalProgressBar* progress_bar) {
  const whisper_sampling_strategy strategy =
    options.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

  whisper_full_params params = whisper_full_default_params(strategy);
  params.n_threads = options.threads;
  params.translate = options.task == Task::kTranslate;
  params.no_timestamps = true;
  params.print_progress = false;
  params.print_realtime = false;
  params.print_timestamps = false;
  params.print_special = false;
  params.progress_callback = whisper_progress_callback;
  params.progress_callback_user_data = progress_bar;

  if (strategy == WHISPER_SAMPLING_BEAM_SEARCH) {
    params.beam_search.beam_size = static_cast<int>(options.beam_size);
  } else {
    params.greedy.best_of = 1;
  }

  if (!whisper_is_multilingual(context)) {
    params.detect_language = false;
    params.language = "en";
    return params;
  }

  if (language_is_auto(options.language)) {
    params.detect_language = true;
    params.language = nullptr;
    return params;
  }

  if (whisper_lang_id(options.language.c_str()) < 0) {
    throw std::invalid_argument("unsupported language '" + options.language + "'");
  }

  params.detect_language = false;
  params.language = options.language.c_str();
  return params;
}

std::string resolve_detected_language(whisper_context* context,
                                      const WhisperRunOptions& options) {
  if (!whisper_is_multilingual(context)) {
    return "en";
  }

  const int language_id = whisper_full_lang_id(context);
  if (language_id >= 0) {
    const char* language = whisper_lang_str(language_id);
    if (language != nullptr) {
      return language;
    }
  }

  if (language_is_auto(options.language)) {
    return {};
  }

  const int explicit_language_id = whisper_lang_id(options.language.c_str());
  if (explicit_language_id >= 0) {
    const char* language = whisper_lang_str(explicit_language_id);
    if (language != nullptr) {
      return language;
    }
  }

  return options.language;
}

}  // namespace

std::string task_to_string(const Task task) {
  return task == Task::kTranslate ? "translate" : "transcribe";
}

WhisperRunResult run_whisper_file(const WhisperRunOptions& options) {
  const std::vector<float> audio = decode_audio_file(options.input_path);
  if (audio.empty()) {
    throw std::runtime_error("decoded audio is empty");
  }

  WhisperContextPtr context = load_context(options);
  TerminalProgressBar progress_bar(options.show_progress);
  whisper_full_params params = build_full_params(context.get(), options, &progress_bar);

  if (whisper_full(context.get(), params, audio.data(), static_cast<int>(audio.size())) != 0) {
    progress_bar.finish();
    throw std::runtime_error("whisper.cpp inference failed");
  }
  progress_bar.finish();

  std::string full_text;
  const int segment_count = whisper_full_n_segments(context.get());
  for (int segment_index = 0; segment_index < segment_count; ++segment_index) {
    const char* segment_text = whisper_full_get_segment_text(context.get(), segment_index);
    if (segment_text != nullptr) {
      append_segment_text(full_text, segment_text);
    }
  }

  return WhisperRunResult{
    resolve_detected_language(context.get(), options),
    trim(full_text),
  };
}

}  // namespace audex
