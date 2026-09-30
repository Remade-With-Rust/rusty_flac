# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
