# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- *(encode)* 32-bit encode (sample-size code 7): channels coded independently
  with CONSTANT, LPC (orders up to the level's cap, 15-bit coefficients),
  FIXED orders 0..=4 or VERBATIM. Predictions run in i64, and a candidate is
  used only when every residual fits 30 bits. LPC windows are Q30, and levels
  5 and 8 add libFLAC-style partial (and, at 8, punchout) Tukey windows; every
  window's predictor at every order is ranked by the exact residual energy of
  its quantized coefficients. On a 384 kHz / 32-bit track, level 8 is 28 %
  smaller than FIXED-only coding, 7.6 % smaller than `flac -8 --no-mid-side`
  and 3.4 % smaller than `flac -8` (which also uses stereo). 8/16/24-bit
  output is unchanged.

## [0.3.1](https://github.com/Remade-With-Rust/rusty_flac/compare/v0.3.0...v0.3.1) - 2026-11-11

### Fixed

- *(decode)* 32-bit stereo streams with a side channel (left/side, side/right,
  mid/side, as `flac` writes them) no longer panic or decode to wrong samples:
  the 33-bit side channel is decoded and undone in i64. Before this,
  `flac -5`/`-8` 32-bit stereo files decoded silently wrong.
  ([#20](https://github.com/Remade-With-Rust/rusty_flac/pull/20), by @benedictjohannes)

## [0.3.0](https://github.com/Remade-With-Rust/rusty_flac/compare/v0.2.2...v0.3.0) - 2026-11-11

Speed and compression release. Every speed change is byte-identical (pinned
`stream_identity` digest, corpus round-trip, and the reference `flac` 1.5.0
decoding every stream back to its source samples); the two output changes
are listed under Changed.

### Changed

- *(encode)* LPC coefficients are 15 bits (was 14), the format maximum and
  libFLAC's choice above 16 bits. Level 8: 96 kHz/24 -3.2%, 384 kHz/24 -4.4%,
  48 kHz piano -0.45%; corpus total -3.2% (worst file +0.02%). Part of the
  high-rate gap in [#22](https://github.com/Remade-With-Rust/rusty_flac/issues/22).
- *(encode)* Levels 0-2 analyse one window instead of two: 17-21% faster at
  level 0 for +1.3% size there (still 7% under `flac -0`). Levels 3-8 unchanged.

### Performance

- Encode at levels 5/8 executes 48%/52% fewer instructions than 0.2.2 at
  44.1 kHz and 46%/46% fewer at 96 kHz (callgrind, AVX2 host); wall time
  -39%/-44% and -40%/-33%.
- Decode executes 9% fewer instructions on `flac -8` streams and 12% fewer on
  `flac -5` streams.
- Encode: Rice partition order capped at 7 (was 8; order 8 changed no corpus
  file); AVX2 LPC residual for sharp predictors (checked per vector) and an
  i32-lane residual where the sums provably fit; u32-lane Rice sums and a
  bounds-check-free partition merge; warm-started convex Rice parameter
  search; AVX2 twins of the windowing pass and the partition planner;
  autocorrelation lags in groups of eight; Rice runs written 32 bits at a
  time (BMI2 twin); u32-lane fixed-order sums; single-pass stereo channel
  construction and deinterleave; packed MD5 interleave and three-operation
  MD5 F/G.
- Decode: LPC restore unrolled by four with BMI2 shifts and a 128-bit window
  for orders 5-8; mask-free Rice code assembly; Rice refill threshold k + 5.

## [0.2.2](https://github.com/Remade-With-Rust/rusty_flac/compare/v0.2.1...v0.2.2) - 2026-10-24

### Other

- *(decode)* 2.1-3.1x faster decode; faster than symphonia 0.5.5 on every file
- 2,081 active installs
- bump rusty_alloc-api to =2.2.6 ([#16](https://github.com/Remade-With-Rust/rusty_flac/pull/16))
- Merge branch 'main' into chore/release-plz-0-5-139
- 1,548 active installs

## [0.2.1](https://github.com/Remade-With-Rust/rusty_flac/compare/v0.2.0...v0.2.1) - 2026-10-08

### Other

- bump rusty_alloc-api to =2.2.5 ([#14](https://github.com/Remade-With-Rust/rusty_flac/pull/14))
- 1,358 active installs

## [0.2.0](https://github.com/Remade-With-Rust/rusty_flac/compare/v0.1.3...v0.2.0) - 2026-09-30

Microcontroller release: the encoder on an ESP32-S3 (240 MHz, `no_std`) goes
from ~124 ms to ~15 ms per 4,096-sample block at level 0. Peak heap drops
from 193–209 KB to 124 KB for an 8,192-sample stream, which `main` after
0.1.3 could not allocate on a 256 KB heap. Host speed is unchanged to 8 %
faster.

### Changed (output bytes)

- The LPC autocorrelation is exact integer arithmetic over Q15 windows,
  normalised so no lag sum can overflow `i64`. **Encoded bytes differ from
  0.1.x**: the corpus total moves by ≤ 0.0001 % at every level (47 real files,
  98.7 MB, plus 24-bit mixes; every stream decodes back bit-exact). std and
  `libm` builds now produce the same bytes on that whole corpus. Pin the same
  version wherever two builds must agree byte for byte.

### Added

- `Encoder::finish_and_reset`: close a stream and start the next on the same
  encoder, keeping its buffers and window cache. The bytes are identical to
  a fresh encoder's.
- `examples/corpus_size.rs` (the compression gate over real audio),
  `tests/peak_heap.rs` (per-stream peak-heap ceilings, in bytes),
  `tests/stream_identity.rs` (a pinned output digest).

### Performance (byte-identical to the new output)

- Full-block windows from a static table; short tail windows cached across
  streams.
- Rice-sum rows sized to the residual's top bit and unrolled per width; the
  windowed-product and Rice-sum scratch share one buffer, freed before it
  grows.
- 32-bit arithmetic wherever it is provably exact on a 32-bit core: the
  fixed-order sums, the Rice sums, the LPC residual (bounded by the block's
  own peak), and a 32-bit bit-writer path for Rice codewords.
- A paired-lag scalar autocorrelation kernel, and a register-blocked AVX2
  integer kernel on x86.
- On the ESP32-S3 these cut retired instructions by 22 % beyond the integer
  autocorrelation itself (exact counts per commit in
  `docs/plans/esp32-encoder-cost.md` §7.6).

### Fixed

- `main` after 0.1.3 was not rustfmt- or clippy-clean, and the `no_std` unit
  tests did not compile.

## [0.1.3](https://github.com/Remade-With-Rust/rusty_flac/compare/v0.1.2...v0.1.3) - 2026-09-17

### Added

- no_std + alloc support behind the std feature, with an optional pure-Rust libm ([#8](https://github.com/Remade-With-Rust/rusty_flac/pull/8))

### Other

- 194 active installs
- an In the wild block above the headline ([#6](https://github.com/Remade-With-Rust/rusty_flac/pull/6))

## [0.1.2](https://github.com/Remade-With-Rust/rusty_flac/compare/v0.1.1...v0.1.2) - 2026-08-28

### Other

- bump rusty_alloc-api to =1.1.6 ([#4](https://github.com/Remade-With-Rust/rusty_flac/pull/4))

## [0.1.1](https://github.com/Remade-With-Rust/rusty_flac/compare/v0.1.0...v0.1.1) - 2026-08-28

### Other

- add release-plz so merged dependency bumps actually reach crates.io ([#2](https://github.com/Remade-With-Rust/rusty_flac/pull/2))
- bump rusty_alloc-api to =1.1.4 ([#1](https://github.com/Remade-With-Rust/rusty_flac/pull/1))
