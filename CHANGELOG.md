# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4](https://github.com/Remade-With-Rust/rusty_flac/compare/v0.1.3...v0.1.4) - 2026-09-20

### Other

- 321 active installs
- *(encode)* reuse one frame BitWriter across the whole stream
- *(encode)* reserve the window-estimate Vec for every window
- *(encode)* borrow subframe samples instead of cloning them
- *(encode)* reuse the Rice-parameter buffers across plans too
- *(encode)* move the phase-1 window estimates into realize_arm
- *(encode)* reuse the autocorrelation buffers; drop the scratch macro
- *(encode)* own the partition-sum scratch on the encoder, reuse it
- *(encode)* size the MD5 chunk buffer to the block, not to CHUNK_FRAMES
- *(encode)* reuse Rice-parameter buffers across partition levels

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
