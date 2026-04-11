/******************************************************************************
 * Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
 * Licensed under the Apache License, Version 2.0
 ******************************************************************************/
#pragma once

// Adapted from FlashAttention's static_switch.h (Tri Dao, 2024)

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)                \
  [&] {                                             \
    if (HEADDIM == 64) {                            \
      constexpr static int kHeadDim = 64;           \
      return __VA_ARGS__();                         \
    } else if (HEADDIM == 128) {                    \
      constexpr static int kHeadDim = 128;          \
      return __VA_ARGS__();                         \
    } else {                                        \
      constexpr static int kHeadDim = 256;          \
      return __VA_ARGS__();                         \
    }                                               \
  }()

#define LANDMARKS_SWITCH(M, ...)                    \
  [&] {                                             \
    if (M <= 32) {                                  \
      constexpr static int kLandmarks = 32;         \
      return __VA_ARGS__();                         \
    } else {                                        \
      constexpr static int kLandmarks = 64;         \
      return __VA_ARGS__();                         \
    }                                               \
  }()
