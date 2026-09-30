# What the encoder costs on an ESP32-S3, and where the cost is

Written 2026-09-30 from a board run in the Janus project (`rusty_esp_audio`,
firmware `xiao-s3-sense-hal-pdm`, row X1 of its killing-C plan). Every number
here was measured on silicon that day or read out of the linked image;
what is inferred says so. The point of the note is the ranked list in §4.

## 1. The setup

- Seeed XIAO ESP32-S3 Sense at 240 MHz, `no_std` + `alloc` on esp-hal 1.2,
  `esp-alloc` heap of 256 KB in internal RAM, no operating system.
- `rusty_flac` **0.1.3 from crates.io** with the `libm` feature, as
  `rusty_esp_audio-core`'s `flac` feature pins it. The local `main` (nine
  `perf(encode)` commits ahead of 0.1.3) was measured once too, §3.
- Input: 16 kHz mono i16 from the board's own PDM microphone in a quiet room
  (peak about −40 dBFS). Each row of the table encodes one complete stream
  through `encode_pcm16` (`Encoder::new` → `push_s16le_bytes` → `finish`):
  the 512 and 4096 rows are the first `samples` of the capture; the 8192 rows
  are **the first 4,096 samples twice over**, so that the two-block stream
  differs from the one-block stream by exactly one block of identical
  content and the subtraction below is exact. (A first version used the
  next 4,096 samples instead, and the subtraction moved with the audio.)
- Times are `esp_hal::time::Instant` around the call, nothing else running,
  nothing printed inside the timed region. Two runs of the same firmware
  agree within 2 %; the 512-sample row varies more (19–30 ms) with content.
- Heap peak is `esp-alloc`'s high-water mark (`internal-heap-stats`) above
  the level before the first encode, so it is the encoder's own.

On this chip an `f64` operation is a call: the S3 has a single-precision FPU
and no `f64` hardware, and esp-hal's linker script resolves `__adddf3`,
`__muldf3`, `__divdf3` and the rest to the mask ROM's libgcc
(`compiler_builtins`' copies are discarded at link). `f32` add and multiply
are instructions; `f32` divide is also a call (`__divsf3`).

## 2. The numbers (0.1.3)

| level | samples | time | µs / sample | FLAC bytes | heap peak |
|---:|---:|---:|---:|---:|---:|
| 0 | 512 | 28.0 ms | 54.6 | 307 | 84,744 |
| 0 | 4096 (A) | 123.6 ms | 30.1 | 2,189 | 172,292 |
| 0 | 8192 (A+A) | 183.4 ms | 22.3 | 4,336 | 192,772 |
| 5 | 4096 (A) | 146.3 ms | 35.7 | 2,189 | 192,772 † |
| 5 | 8192 (A+A) | 229.2 ms | 27.9 | 4,336 | 192,772 † |
| 8 | 4096 (A) | 169.4 ms | 41.3 | 2,189 | 192,772 † |
| 8 | 8192 (A+A) | 275.4 ms | 33.6 | 4,336 | 192,772 † |
| 5 | 8000 | 281.1 ms | 35.1 | 3,973 | 192,772 † |

† the high-water mark had already been set by the 8192-sample row; these
encodes did not exceed it. On a louder two seconds from the same room the
mark reached **209,164 B** for the 8192 rows: the peak moves with the
content, and any margin has to be read against that number, not 192,772.

Two facts fall out of the pairs of rows, and they are the whole story:

**One more block of the same content costs 59.8 / 82.9 / 106.0 ms at levels
0 / 5 / 8** (t(A+A) − t(A)): 14.6, 20.2 and 25.9 µs per sample. Real time
at 16 kHz mono is 62.5 µs per sample, so the steady state is 2.4–4.3×
faster than real time on this content; louder content costs more (one
block of the louder capture cost 100 ms at level 0, 24.5 µs per sample).
Stereo at 48 kHz would need about 10 µs per sample and is out of reach on
this chip today.

**Everything that is not a block costs 63–64 ms, whatever the level**
(2 t(A) − t(A+A): 63.8, 63.4 and 63.4 ms). A one-block stream pays it
once; an 8,000-sample stream pays it **twice** — its last block is 3,904
samples, a size the window cache has not seen — which is why the 8,000-sample
row (281 ms) is slower than the 8,192-sample row (229 ms) despite having 192
fewer samples. Level-independent and size-dependent means this is
`WindowCache::ensure` → `tukey_window`: two windows of `n` `f64` values,
about 0.7 n of them through `libm::cos`, rebuilt for every stream (the cache
lives inside `encode_stream`) and again for the short tail. At 4096 that is
about 2,870 soft-float cosines. For a device that encodes in chunks of a
second or two — the shape `rusty_esp_audio` uses — the windows are a third
of the encode.

The 512-sample row is the same thing at small `n`: 19–31 ms across the runs
for 32 ms of audio, most of it fixed cost. That is what Janus's A3 row measured last
month as "177.8 µs per sample" and attributed to the encoder in general; it
was mostly the windows of a 512-sample stream.

## 3. `main` against 0.1.3 on the same board

The nine `perf(encode)` commits since 0.1.3 reuse scratch across calls and
cut allocation *count*. On the chip they change the *peak*:

| | 512 | 4096 | 8192 |
|---|---:|---:|---:|
| 0.1.3 peak heap | 84,744 | 172,292 | 192,772 |
| `main` peak heap | 64,000 | **214,396** | **fails**: a 16,384-byte allocation is refused, in a heap that had 243 KB free when the encode began |
| `main` time, level 0 | 20.8 ms | 123.4 ms | — |

(`main`'s rows were taken before the A+A change, with the capture's own
second block; the 512 and 4096 rows are unaffected by that.)

Time is unchanged where both ran. The 512-sample peak fell (the MD5 chunk
buffer sized to the block, commit `cfc2b45`), the 4096 peak rose by 42 KB,
and an 8,192-sample stream that 0.1.3 encodes in 193 KB does not fit `main`
in 243 KB. Buffers that are reused across subframes and frames are alive
for the whole stream, and now coexist with the largest transient set instead
of taking turns with it; the reserved `BitWriter` (raw ceiling of the first
block) and the reserved window-estimate `Vec` add to the same peak. None of
this shows on a host with gigabytes; an allocation-count gate did not catch
it because the count went down. **A peak-bytes number per encode belongs
next to the allocation count in the ledger**, and the question for the
series is which commit(s) moved it — the same counting allocator that
produced the 81 → 26 figure can report a high-water mark per commit.

## 4. Opportunities, ranked by bytes-and-milliseconds per unit of risk

Each says what changes, what the chip gains, and what gate it needs. "Byte-
identical" means the FLAC stream is unchanged; the lossless round trip and
the host–chip identity hold either way, but a stream change needs the
corpus compression gate before it lands.

1. **Stop rebuilding the windows.** Make the `BLOCK_SIZE` windows a table
   computed once and kept: computed at build time by a `build.rs` with the
   same `libm::cos`, they are bit-identical to what `tukey_window` produces
   at run time, so the stream is byte-identical. Only the tapers are not
   `1.0`: 2 × 1024 + 2 × 410 values, 23 KB of flash if stored as `f64`, and a
   `&'static [f64]` costs no RAM at all — which also takes **64 KB off the
   peak heap** (two `Vec<f64>` of 4096) and is the single largest item in
   §3's table. The short tail block still computes its own window (55 ms at
   3,904 samples); §5 says how a caller avoids it, and a persistent
   two-entry cache handed in by the caller would remove it for repeated
   chunk sizes. Gain on this board: −63 ms per stream, −64 KB peak.
   Byte-identical.

2. **Autocorrelation in integers.** The per-block cost rises 5.6 ms per four
   extra LPC lags at both level steps (0 → 5 and 5 → 8), which is 1.4 µs per
   sample per lag; each lag is two `f64` operations per window per sample,
   so this puts one soft-float operation at roughly 0.35 µs — 84 cycles —
   *if* the lag cost is all autocorrelation, which is an inference, not a
   measurement. On that inference, windowing plus autocorrelation is about
   24 `f64` operations per sample at level 0 (57 % of the block cost) and 56
   at level 8 (76 %). A fixed-point window (Q31) times the sample, and the
   lag dot products in `i64`, is exact in integers, keeps host and chip
   identical by construction, and on the S3 is the shape of the PIE
   multiply-accumulate instructions the Janus DSP work already uses.
   Levinson–Durbin stays in `f64` on the `order + 1` sums: negligible. It is
   a stream change — the window is quantized — so it needs the corpus gate,
   and measuring first is cheap: count `f64` operations per sample on the
   host with a wrapping type (the count is the same on the chip), then time
   one block on the board with the autocorrelation stubbed to zero to bound
   its share exactly.

3. **The 64-bit divisions.** `rice_bits_estimate`, `realize_arm` and
   `encode_stream` each hold a `__udivdi3` site (mask ROM). Per partition
   candidate, not per sample; probably small, and the same counting
   instrument as in 2 says exactly how small.

4. **Not worth it here:** `log2` in order selection (≤ 12 calls per window
   per block), `exp2`/`round` in `quantize_lpc`, `sqrt` — all per block, not
   per sample; and replacing the ROM's soft-float with `compiler_builtins`'
   (same speed class, same work). An `f32` autocorrelation on the FPU is
   tempting and wrong: a 24-bit mantissa cannot carry sums over a 4,096-sample
   block of 16-bit audio.

## 5. What the caller can do meanwhile (for `rusty_esp_audio`)

Chunk in whole blocks. `FlacEncoder` in `rusty_esp_audio-core` finishes a
stream per chunk; a chunk that is a multiple of 4,096 samples pays one
window build, not two. 8,192 samples (512 ms at 16 kHz) is the natural
size; 8,000 was arbitrary. And size the heap from §2: 193 KB peak for an
8,192-sample stream on 0.1.3, plus whatever the caller holds.

## 6. How to reproduce

```sh
# In the Janus umbrella, with the XIAO on COM4 and espino built:
cd rusty_esp_audio/firmware/xiao-s3-sense-hal-pdm
cargo build --release                                   # crates.io rusty_flac
cargo build --release --config 'patch.crates-io.rusty_flac.path="F:/coding/rusty_flac"'   # local main
espino flash   --board xiao-esp32s3-sense --port COM4 --app <ELF>
espino monitor --board xiao-esp32s3-sense --port COM4 --app <ELF> --expect "== DONE ==" --timeout 60 > capture.txt
grep FLACTIME capture.txt
```

The `FLACTIME` lines are §2's rows; `FLAC begin` is the 8,000-sample row and
carries the heap free before and after. On the host, `cargo run -p
rusty_esp_audio-esp --features std,flac --example chip_capture_check --
capture.txt` checks that the chip's stream equals the host's, byte for byte.
The functions that reach the ROM were found by disassembling the image
(`xtensa-esp32s3-elf-objdump -d`) and matching every `l32r` literal against
the kept relocations of a link made with `--emit-relocs`; on Xtensa a call
goes through the literal pool, so an address-range attribution finds
nothing.

## 7. Executed — 2026-09-30, rusty_flac `main` after `fdaf5d6`

Every item above is done or priced. Numbers below come from the same XIAO
ESP32-S3 at 240 MHz, `no_std` + `libm`, esp-alloc 256 KB, opt-level `s`.
A bench firmware fed **fixed PCM** instead of the live microphone, so the
two builds in each comparison encode identical samples. `quiet` is
coloured noise near −40 dBFS with a 50 Hz hum (the §1 room); `loud` is
three tones plus noise near −6 dBFS. Peak is a counting wrapper around
`esp_alloc::HEAP` (requested bytes). Every chip stream's FNV digest equals
the host's, all 16 rows.

### 7.1 Where the chip is now

| quiet, 16 kHz mono | `main` (62761b0) | now |
|---|---:|---:|
| 4096 samples, L0 | 139.8 ms, peak 231,752 B | **21.5 ms, peak 102,728 B** |
| 4096, L5 / L8 | — | 24.6 / 27.8 ms |
| 8192 (A+A), L0 | allocation abort | **42.6 ms, peak 123,208 B** |
| 8192, L8 | allocation abort | 55.1 ms |
| loud 8192, L8 | allocation abort | 71.2 ms, peak 123,444 B |

- **One more block** (t(A+A) − t(A)) now costs 21.1 / 24.2 / 27.3 ms at
  levels 0 / 5 / 8 on quiet content. That is 5.1–6.7 µs per sample against
  real time's 62.5, so 9–12× faster than real time (§2 had 2.4–4.3×). Loud
  content at level 8 costs 8.6 µs per sample.
- **Everything that is not a block** (2t(A) − t(A+A)) is 0.4 ms, down from
  63.8 ms.
- **Peak heap** is flat across levels and nearly flat across content. The
  worst case is full-scale white noise at 139.7 KB (host instrument), so
  **size the heap for 140 KB plus what the caller holds**.

### 7.2 Item by item

1. **Windows** (`e242aa3`, then Q15 in `58c9437`). The `BLOCK_SIZE` tapers
   are a static table: now 5.7 KB of `u16` flash, for every build. The flat
   middle is neither stored nor multiplied. −63 ms per stream and −64 KB of
   peak, as predicted.
   - A short tail block still builds its own tapers, about 50 ms per
     3,904-sample tail with soft-float `cos`.
   - The new `Encoder::finish_and_reset(&mut self)` (`c967698`, cache fix
     `c5c174f`) keeps those tapers and every scratch buffer across streams.
     An 8,000-sample chunk costs 90.4 ms the first time and **39.2 ms**
     after that at L0, with byte-identical output. The price is about 73 KB
     held between chunks.
2. **Autocorrelation in integers** (`58c9437`). The measured share on the
   chip was 57 % at L0 and 78 % at L8, confirming §4.2's inference. Windows
   are Q15 and windowed samples are normalised per block, so that
   |x| ≤ 2^b with 2b + ⌈log2 n⌉ ≤ 62. **Every lag sum then fits `i64` by
   construction**; a Q31 window with raw i64 sums, as first written above,
   would overflow. The lag sums are `i32 × i32 → i64`, which is native
   `mull`/`mulsh` on Xtensa.
   - Autocorrelation went from 32.0 to about 5 ms per block at L0, and the
     whole encode got 2–3× faster.
   - It changes the stream. The corpus gate (47 real files, 98.7 MB of raw
     PCM, plus three 24-bit mixes, all decoded back bit-exact) moved
     +0.00002 % / +0.00008 % / +0.00004 % at L0/L5/L8, with the worst file
     at +0.002 %.
   - The host is not slower: 0.955× at L5 on music, 15/16 paired wins.
   - std and `libm` builds now produce identical bytes on every corpus file.
3. **64-bit divisions**. The disassembly still shows exactly the three
   `__udivdi3` sites named above. Each runs once per call, one or two calls
   per arm per block, which is about 1 µs per 21 ms block. Left alone.
4. **Not done, as advised.** On `f32`: libFLAC ran a float autocorrelation
   for years, so it is a compression question for the corpus gate, not a
   correctness one. It is moot now.

### 7.3 Section 3's regression, and what else the profile found

- **Peak.** `wprod` (the e820e5e regression) now shares one buffer with the
  Rice-sum rows (`df00a64`); the two never run at once.
- **Rice rows.** They stop at the residual's top bit (`1f00290`): 63,488 →
  about 20 KB on quiet audio, and 31 → about 11 u64 adds per residual
  sample on the chip.
- **Grow-before-free.** Buffers are freed before they grow (`df00a64`),
  because a `realloc` held both copies at the peak for loud L8.
- **Peak ceilings.** `tests/peak_heap.rs` gates per-stream peaks in bytes,
  next to the allocation count. The 0.1.3 → `main` attribution per commit
  is in `b3791cf`'s parent log.
- **32-bit arithmetic** (`1253475`) in the scalar fixed-order and Rice sums,
  byte-identical: 25.4 → 21.5 ms at L0.
- **Tried and reverted.** A bit-sliced constant-cost Rice kernel
  (`784fc7b`, reverted `0849c96`) measured 21.5 → 23.3 ms, so it lost.

What an L0 quiet block spends now (CCOUNT, with its overhead included):
autocorrelation 5.4 ms, the two partition plans 4.7 + 4.9 ms, fixed-order
estimate 3.0 ms, subframe writing 2.1 ms, MD5 1.4 ms. There is no single
stage left above 25 %.

### 7.4 For `rusty_esp_audio` (section 5, updated)

- **Pin the next rusty_flac release on both sides.** The stream changed in
  `58c9437`, so the chip/host byte-identity check needs the same version on
  the laptop.
- **Chunk in whole blocks** (8,192 samples), or keep one `rusty_flac::Encoder`
  and call `finish_and_reset` per chunk instead of `FlacEncoder` creating
  one per chunk. Either removes the tail-window cost; the second also stops
  re-allocating about 73 KB of scratch per chunk.
- **Heap.** 124 KB peak for an 8,192-sample stream (140 KB worst case),
  against 193–209 KB on 0.1.3.
- The A3 row's "177.8 µs per sample" was mostly the windows of a
  512-sample stream (§2). That stream now takes 13.7 ms, about 27 µs per
  sample, and is still mostly fixed cost: MD5, STREAMINFO and a
  short-block window build.

### 7.5 Reproducing 7.1 without the microphone

The host twins are:

- `cargo test --release --no-default-features --features libm --test peak_heap -- --nocapture`
  (add `RUSTY_FLAC_PEAK_DUMP=1` to list the allocations live at each peak);
- `tests/stream_identity.rs`, a pinned digest over 384 encodes;
- `cargo run --release --example corpus_size -- <wav/flac>...`, the
  compression gate.

The chip numbers came from a scratch firmware: the §6 firmware's profile,
esp-alloc without `global-allocator` behind a counting wrapper, and fixed
PCM through `include_bytes!`. That firmware lives outside this repo. On the
Janus firmware itself, rerun §6 against this commit.
