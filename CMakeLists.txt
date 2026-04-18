cmake_minimum_required(VERSION 3.20)
project(Audex LANGUAGES C CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavcodec
  libavformat
  libavutil
  libswresample
)

find_package(whisper CONFIG QUIET)

if(NOT TARGET whisper)
  include(FetchContent)

  set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    whispercpp
    GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
    GIT_TAG v1.8.4
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    GIT_SUBMODULES_RECURSE TRUE
  )

  FetchContent_MakeAvailable(whispercpp)
endif()

add_executable(audex
  src/audio.cpp
  src/audio.hpp
  src/main.cpp
  src/whisper.cpp
  src/whisper.hpp
)

target_compile_features(audex PRIVATE cxx_std_17)

target_link_libraries(audex
  PRIVATE
    whisper
    PkgConfig::FFMPEG
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(audex PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS audex RUNTIME DESTINATION bin)
