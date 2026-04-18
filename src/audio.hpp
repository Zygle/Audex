#pragma once

#include <string>
#include <vector>

namespace audex {

constexpr int kSampleRate = 16000;

std::vector<float> decode_audio_file(const std::string& path);

}  // namespace audex
