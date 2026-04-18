#include "audio.hpp"

#include <cerrno>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

namespace audex {
namespace {

struct FormatContextDeleter {
  void operator()(AVFormatContext* context) const {
    if (context != nullptr) {
      avformat_close_input(&context);
    }
  }
};

struct CodecContextDeleter {
  void operator()(AVCodecContext* context) const {
    if (context != nullptr) {
      avcodec_free_context(&context);
    }
  }
};

struct PacketDeleter {
  void operator()(AVPacket* packet) const {
    if (packet != nullptr) {
      av_packet_free(&packet);
    }
  }
};

struct FrameDeleter {
  void operator()(AVFrame* frame) const {
    if (frame != nullptr) {
      av_frame_free(&frame);
    }
  }
};

struct SwrContextDeleter {
  void operator()(SwrContext* context) const {
    if (context != nullptr) {
      swr_free(&context);
    }
  }
};

std::string ffmpeg_error_text(const int error_code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(error_code, buffer, sizeof(buffer));
  return buffer;
}

void throw_if_ffmpeg_error(const int error_code, const std::string& context) {
  if (error_code < 0) {
    throw std::runtime_error(context + ": " + ffmpeg_error_text(error_code));
  }
}

void append_converted_frame(SwrContext* resampler,
                            const int input_sample_rate,
                            AVFrame* frame,
                            std::vector<float>& output) {
  const int estimated_output_samples =
    av_rescale_rnd(swr_get_delay(resampler, input_sample_rate) + frame->nb_samples,
                   kSampleRate,
                   input_sample_rate,
                   AV_ROUND_UP);

  std::vector<float> converted(static_cast<std::size_t>(estimated_output_samples));
  uint8_t* output_planes[] = {
    reinterpret_cast<uint8_t*>(converted.data()),
  };

  const uint8_t** input_planes = const_cast<const uint8_t**>(frame->extended_data);
  const int written_samples =
    swr_convert(resampler,
                output_planes,
                estimated_output_samples,
                input_planes,
                frame->nb_samples);

  throw_if_ffmpeg_error(written_samples, "failed to resample audio frame");
  converted.resize(static_cast<std::size_t>(written_samples));
  output.insert(output.end(), converted.begin(), converted.end());
}

void flush_resampler(SwrContext* resampler,
                     const int input_sample_rate,
                     std::vector<float>& output) {
  while (true) {
    const int delayed_samples =
      av_rescale_rnd(swr_get_delay(resampler, input_sample_rate),
                     kSampleRate,
                     input_sample_rate,
                     AV_ROUND_UP);

    if (delayed_samples <= 0) {
      return;
    }

    std::vector<float> converted(static_cast<std::size_t>(delayed_samples));
    uint8_t* output_planes[] = {
      reinterpret_cast<uint8_t*>(converted.data()),
    };

    const int written_samples =
      swr_convert(resampler, output_planes, delayed_samples, nullptr, 0);

    throw_if_ffmpeg_error(written_samples, "failed to flush resampler");

    if (written_samples == 0) {
      return;
    }

    converted.resize(static_cast<std::size_t>(written_samples));
    output.insert(output.end(), converted.begin(), converted.end());
  }
}

void drain_decoder(AVCodecContext* decoder_context,
                   SwrContext* resampler,
                   AVFrame* frame,
                   std::vector<float>& output) {
  while (true) {
    const int receive_status = avcodec_receive_frame(decoder_context, frame);

    if (receive_status == AVERROR(EAGAIN) || receive_status == AVERROR_EOF) {
      return;
    }

    throw_if_ffmpeg_error(receive_status, "failed to decode audio frame");
    append_converted_frame(resampler, decoder_context->sample_rate, frame, output);
    av_frame_unref(frame);
  }
}

}  // namespace

std::vector<float> decode_audio_file(const std::string& path) {
  AVFormatContext* raw_format_context = nullptr;
  throw_if_ffmpeg_error(
    avformat_open_input(&raw_format_context, path.c_str(), nullptr, nullptr),
    "failed to open input audio");
  std::unique_ptr<AVFormatContext, FormatContextDeleter> format_context(raw_format_context);

  throw_if_ffmpeg_error(avformat_find_stream_info(format_context.get(), nullptr),
                        "failed to read input stream information");

  const int audio_stream_index = av_find_best_stream(format_context.get(),
                                                     AVMEDIA_TYPE_AUDIO,
                                                     -1,
                                                     -1,
                                                     nullptr,
                                                     0);
  throw_if_ffmpeg_error(audio_stream_index, "failed to find an audio stream");

  const AVStream* audio_stream =
    format_context->streams[static_cast<std::size_t>(audio_stream_index)];
  const AVCodec* decoder = avcodec_find_decoder(audio_stream->codecpar->codec_id);
  if (decoder == nullptr) {
    throw std::runtime_error("failed to find a decoder for the audio stream");
  }

  AVCodecContext* raw_decoder_context = avcodec_alloc_context3(decoder);
  if (raw_decoder_context == nullptr) {
    throw std::runtime_error("failed to allocate decoder context");
  }
  std::unique_ptr<AVCodecContext, CodecContextDeleter> decoder_context(raw_decoder_context);

  throw_if_ffmpeg_error(
    avcodec_parameters_to_context(decoder_context.get(), audio_stream->codecpar),
    "failed to copy codec parameters to decoder");
  throw_if_ffmpeg_error(avcodec_open2(decoder_context.get(), decoder, nullptr),
                        "failed to open audio decoder");

  AVChannelLayout output_layout;
  av_channel_layout_default(&output_layout, 1);

  SwrContext* raw_resampler = nullptr;
  throw_if_ffmpeg_error(
    swr_alloc_set_opts2(&raw_resampler,
                        &output_layout,
                        AV_SAMPLE_FMT_FLT,
                        kSampleRate,
                        &decoder_context->ch_layout,
                        decoder_context->sample_fmt,
                        decoder_context->sample_rate,
                        0,
                        nullptr),
    "failed to configure audio resampler");
  av_channel_layout_uninit(&output_layout);

  std::unique_ptr<SwrContext, SwrContextDeleter> resampler(raw_resampler);
  throw_if_ffmpeg_error(swr_init(resampler.get()), "failed to initialize resampler");

  std::unique_ptr<AVPacket, PacketDeleter> packet(av_packet_alloc());
  std::unique_ptr<AVFrame, FrameDeleter> frame(av_frame_alloc());
  if (packet == nullptr || frame == nullptr) {
    throw std::runtime_error("failed to allocate FFmpeg packet or frame");
  }

  std::vector<float> audio_samples;

  while (true) {
    const int read_status = av_read_frame(format_context.get(), packet.get());
    if (read_status == AVERROR_EOF) {
      break;
    }
    throw_if_ffmpeg_error(read_status, "failed to read audio packet");

    if (packet->stream_index == audio_stream_index) {
      throw_if_ffmpeg_error(avcodec_send_packet(decoder_context.get(), packet.get()),
                            "failed to send packet to decoder");
      drain_decoder(decoder_context.get(), resampler.get(), frame.get(), audio_samples);
    }

    av_packet_unref(packet.get());
  }

  throw_if_ffmpeg_error(avcodec_send_packet(decoder_context.get(), nullptr),
                        "failed to flush decoder");
  drain_decoder(decoder_context.get(), resampler.get(), frame.get(), audio_samples);
  flush_resampler(resampler.get(), decoder_context->sample_rate, audio_samples);

  return audio_samples;
}

}  // namespace audex
