//! In-house **FLAC encoder** — lossless, pure Rust, no FFI.
//!
//! Ported from `rff-codec-flac` (built brick by brick; see that crate's
//! `docs/codec-flac-encoder.md` history). The port keeps the *decisions*
//! byte-identical to the original while replacing the primitives underneath:
//! an accumulator bit writer (was bit-by-bit), table CRCs (was bitwise),
//! cached apodization windows (was cos() per sample per subframe), an exact
//! bottom-up sum-merged Rice partition planner (was a full 15-parameter scan
//! per partition per order), and batched MD5 feeding (was per-sample rows).
//!
//! The encoder buffers the whole stream and emits a complete native FLAC
//! stream from [`Encoder::finish`] — framing, STREAMINFO and MD5 included.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::bitio::BitWriter;
use crate::crc::{crc16, crc8};
use crate::math;

/// Default samples-per-channel per FLAC frame, changed by
/// [`Encoder::set_block_size`]. 4096 is FLAC's usual default; every size is
/// encoded as an explicit 16-bit block size (frame-header block-size code 7).
const BLOCK_SIZE: usize = 4096;
/// The block sizes FLAC allows (STREAMINFO's minimum is 16).
const BLOCK_SIZES: core::ops::RangeInclusive<u32> = 16..=65535;
/// Quantized LPC coefficient precision in bits: 15, the largest the format
/// allows, as libFLAC uses for every depth above 16 bits.
///
/// It was 14. Sharp predictors (high-rate audio, whose top band is nearly
/// empty, reach coefficients of 30-60) lose more to a coarser grid than the
/// extra bit per coefficient costs: at level 8, 96 kHz/24 -3.2% and
/// 384 kHz/24 -4.4%; 48 kHz/16 piano and speech also shrink (-0.45%,
/// -0.13%), 44.1 kHz guitar grows 0.02%. On `flac -8` itself, 14 bits cost
/// 4.4% on the same 96 kHz file.
const LPC_PRECISION: u32 = 15;
/// Highest LPC order searched — subset-compliant.
pub(crate) const LPC_MAX_ORDER: usize = 12;
/// Rice parameters searched. Method 0 (4-bit params) covers k 0..=14; method
/// 1 (Rice2, 5-bit params) extends to k 0..=30 — essential for high-entropy
/// 24-bit residuals, where the optimal parameter sits well above 14.
const RICE_KMAX: usize = 30;
/// Largest parameter expressible in a method-0 (4-bit) partition. 15/31 are
/// the escape codes and are never emitted.
const RICE_KMAX_M0: usize = 14;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Encoder configuration / stream errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// FLAC's channel-assignment field caps independent channels at 8.
    TooManyChannels(u32),
    /// Zero channels.
    NoChannels,
    /// Bits per sample outside the supported 8/16/24/32 set.
    UnsupportedBps(u32),
    /// Sample rate must fit STREAMINFO's 20-bit field and be non-zero.
    BadSampleRate(u32),
    /// push_interleaved got a slice whose length is not a channel multiple.
    RaggedInput,
    /// Block size outside FLAC's 16..=65535.
    BadBlockSize(u32),
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EncodeError::TooManyChannels(c) => write!(f, "flac: {c} channels (max 8)"),
            EncodeError::NoChannels => write!(f, "flac: zero channels"),
            EncodeError::UnsupportedBps(b) => write!(f, "flac: unsupported bit depth {b}"),
            EncodeError::BadSampleRate(r) => write!(f, "flac: bad sample rate {r}"),
            EncodeError::RaggedInput => {
                write!(f, "flac: interleaved length not a channel multiple")
            }
            EncodeError::BadBlockSize(b) => write!(f, "flac: bad block size {b}"),
        }
    }
}

impl core::error::Error for EncodeError {}

/// Wiring-audit counters: every decision path in the encoder counts what it
/// chose, so a corpus run can prove no path is silently dead and no fallback
/// is silently hot. Cheap (a few increments per subframe).
#[derive(Debug, Default, Clone)]
pub struct EncodeStats {
    pub frames: u64,
    /// Chosen subframe kinds, over all written subframes.
    pub sub_constant: u64,
    pub sub_verbatim: u64,
    pub sub_fixed: u64,
    pub sub_lpc: u64,
    /// Stereo channel assignments chosen (stereo streams only).
    pub stereo_independent: u64,
    pub stereo_left_side: u64,
    pub stereo_right_side: u64,
    pub stereo_mid_side: u64,
    /// LPC machinery health.
    pub lpc_quantize_failed: u64,
    pub lpc_levinson_exhausted: u64,
    pub lpc_window_second_won: u64,
    /// Histogram of chosen partition orders (0..=8).
    pub partition_orders: [u64; 9],
    /// Fixed-predictor orders chosen (0..=4).
    pub fixed_orders: [u64; 5],
    /// Subframes that shifted out trailing zero bits (wasted-bits path).
    pub sub_wasted_bits: u64,
}

/// Reusable analysis buffers owned by the encoder: the hot per-subframe
/// scratch is allocated once and reused for the encoder's life instead of
/// once per call. On `std` this is what the former thread-local pools did; on
/// `no_std` (no thread-locals) it is the only reuse mechanism, and it is what
/// makes a per-block encode on a small heap affordable. Cleared, never
/// truncated in a way that changes an output byte.
#[derive(Default)]
struct EncodeScratch {
    /// One word buffer shared by the two largest analysis stages, which never
    /// run at the same time: `autocorrelation`'s windowed products (f64 bit
    /// patterns, one per sample) and `plan_partitions`' Rice sums (one row of
    /// `stride` shifted sums per finest partition, `stride` stopping at the
    /// residual's top bit). Shared, the encoder's peak holds the larger of the
    /// two instead of both.
    words: Vec<u64>,
    /// `autocorrelation` output, lags `0..=max_order`.
    autoc: Vec<f64>,
    /// `plan_partitions` per-level Rice parameters for the two coding methods
    /// (transient within a plan; the winner is copied out).
    ks0: Vec<u32>,
    ks1: Vec<u32>,
}

/// A pure-Rust FLAC encoder. Feed planar or interleaved `i32` samples at the
/// configured bit depth, then [`Encoder::finish`] returns the complete stream.
pub struct Encoder {
    sample_rate: u32,
    channels: usize,
    bps: u32,
    max_lpc_order: usize,
    block_size: usize,
    chans: Vec<Vec<i32>>,
    stats: EncodeStats,
    scratch: EncodeScratch,
    /// LPC windows for the last block size seen; kept across streams by
    /// [`Encoder::finish_and_reset`].
    wins: WindowCache,
}

impl Encoder {
    pub fn new(sample_rate: u32, channels: u32, bits_per_sample: u32) -> Result<Self, EncodeError> {
        if channels == 0 {
            return Err(EncodeError::NoChannels);
        }
        if channels > 8 {
            return Err(EncodeError::TooManyChannels(channels));
        }
        if !matches!(bits_per_sample, 8 | 16 | 24 | 32) {
            return Err(EncodeError::UnsupportedBps(bits_per_sample));
        }
        if sample_rate == 0 || sample_rate >= (1 << 20) {
            return Err(EncodeError::BadSampleRate(sample_rate));
        }
        Ok(Encoder {
            sample_rate,
            channels: channels as usize,
            bps: bits_per_sample,
            max_lpc_order: LPC_MAX_ORDER,
            block_size: BLOCK_SIZE,
            chans: vec![Vec::new(); channels as usize],
            stats: EncodeStats::default(),
            scratch: EncodeScratch::default(),
            wins: WindowCache::default(),
        })
    }

    /// `0..=8`, the ffmpeg/libFLAC-style speed-vs-ratio knob (maps onto the max
    /// LPC order searched).
    pub fn set_compression_level(&mut self, level: u32) {
        // Levels 0-2 analyse one window, not two: the second window's
        // autocorrelation, its estimate and (when it is close) its
        // realization are the cost that made these levels nearly as slow as
        // level 5 while producing identical output to one another.
        self.wins.single = level <= 2;
        self.max_lpc_order = if level <= 2 {
            4
        } else if level <= 5 {
            8
        } else {
            12
        };
    }

    /// Samples per channel in every frame but the last, `16..=65535`
    /// (default 4096). Sizes above 4608 leave FLAC's streamable subset (above
    /// 16384 for sample rates over 48 kHz). Only 4096 reads its LPC windows
    /// from a static table; any other size builds them (16/24-bit and 32-bit
    /// alike) once and keeps them, across streams too with
    /// [`Encoder::finish_and_reset`].
    pub fn set_block_size(&mut self, block_size: u32) -> Result<(), EncodeError> {
        if !BLOCK_SIZES.contains(&block_size) {
            return Err(EncodeError::BadBlockSize(block_size));
        }
        self.block_size = block_size as usize;
        Ok(())
    }

    /// Append interleaved samples (len must be a channel multiple).
    pub fn push_interleaved(&mut self, samples: &[i32]) -> Result<(), EncodeError> {
        let ch = self.channels;
        if samples.len() % ch != 0 {
            return Err(EncodeError::RaggedInput);
        }
        if ch == 1 {
            self.chans[0].extend_from_slice(samples);
            return Ok(());
        }
        let n = samples.len() / ch;
        if ch == 2 {
            // Whole frames through `chunks_exact` into reserved space: no
            // capacity check per sample, one pass. `step_by` over the flat
            // slice walked it once per channel, a sample at a time.
            let [l, r] = &mut self.chans[..] else {
                unreachable!("two channels")
            };
            // One pass over the frames writes both channels' reserved tails.
            let (l0, r0) = (l.len(), r.len());
            l.reserve(n);
            r.reserve(n);
            for ((a, b), f) in l.spare_capacity_mut()[..n]
                .iter_mut()
                .zip(&mut r.spare_capacity_mut()[..n])
                .zip(samples.chunks_exact(2))
            {
                a.write(f[0]);
                b.write(f[1]);
            }
            // SAFETY: the loop initialized the `n` slots past each old length.
            unsafe {
                l.set_len(l0 + n);
                r.set_len(r0 + n);
            }
            return Ok(());
        }
        for (c, chan) in self.chans.iter_mut().enumerate() {
            chan.reserve(n);
            chan.extend(samples[c..].iter().step_by(ch));
        }
        Ok(())
    }

    /// Append interleaved little-endian s16 PCM bytes (the WAV `data` layout)
    /// in one pass — no intermediate i32 buffer needed by the caller.
    pub fn push_s16le_bytes(&mut self, bytes: &[u8]) -> Result<(), EncodeError> {
        let ch = self.channels;
        if bytes.len() % (2 * ch) != 0 {
            return Err(EncodeError::RaggedInput);
        }
        let frames = bytes.len() / (2 * ch);
        for chan in self.chans.iter_mut() {
            chan.reserve(frames);
        }
        match ch {
            1 => {
                self.chans[0].extend(
                    bytes
                        .chunks_exact(2)
                        .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32),
                );
            }
            2 => {
                let (l, r) = self.chans.split_at_mut(1);
                for c in bytes.chunks_exact(4) {
                    l[0].push(i16::from_le_bytes([c[0], c[1]]) as i32);
                    r[0].push(i16::from_le_bytes([c[2], c[3]]) as i32);
                }
            }
            _ => {
                for row in bytes.chunks_exact(2 * ch) {
                    for (c, chan) in self.chans.iter_mut().enumerate() {
                        chan.push(i16::from_le_bytes([row[c * 2], row[c * 2 + 1]]) as i32);
                    }
                }
            }
        }
        Ok(())
    }

    /// Append interleaved little-endian f32 PCM bytes, quantized onto this
    /// encoder's `bits_per_sample` grid (round-half-away, clamped) — the
    /// float-input convention shared with ffmpeg's flac encoder.
    pub fn push_f32le_bytes(&mut self, bytes: &[u8]) -> Result<(), EncodeError> {
        let ch = self.channels;
        if bytes.len() % (4 * ch) != 0 {
            return Err(EncodeError::RaggedInput);
        }
        let scale = (1i64 << (self.bps - 1)) as f32;
        let frames = bytes.len() / (4 * ch);
        for chan in self.chans.iter_mut() {
            chan.reserve(frames);
        }
        let quant = |c: &[u8]| -> i32 {
            let s = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            math::roundf(s * scale).clamp(-scale, scale - 1.0) as i32
        };
        match ch {
            1 => self.chans[0].extend(bytes.chunks_exact(4).map(quant)),
            2 => {
                let (l, r) = self.chans.split_at_mut(1);
                for c in bytes.chunks_exact(8) {
                    l[0].push(quant(&c[0..4]));
                    r[0].push(quant(&c[4..8]));
                }
            }
            _ => {
                for row in bytes.chunks_exact(4 * ch) {
                    for (c, chan) in self.chans.iter_mut().enumerate() {
                        chan.push(quant(&row[c * 4..c * 4 + 4]));
                    }
                }
            }
        }
        Ok(())
    }

    /// Append per-channel (planar) samples; all planes must be equal length.
    pub fn push_planar(&mut self, planes: &[&[i32]]) -> Result<(), EncodeError> {
        if planes.len() != self.channels {
            return Err(EncodeError::RaggedInput);
        }
        let n = planes[0].len();
        if planes.iter().any(|p| p.len() != n) {
            return Err(EncodeError::RaggedInput);
        }
        for (chan, plane) in self.chans.iter_mut().zip(planes) {
            chan.extend_from_slice(plane);
        }
        Ok(())
    }

    /// Encode all buffered samples into a complete native FLAC stream.
    pub fn finish(mut self) -> Vec<u8> {
        self.encode_stream()
    }

    /// Like [`Encoder::finish`], but also returns the wiring-audit counters.
    pub fn finish_with_stats(mut self) -> (Vec<u8>, EncodeStats) {
        let out = self.encode_stream();
        let stats = core::mem::take(&mut self.stats);
        (out, stats)
    }

    /// Encode all buffered samples into a complete FLAC stream, then start the
    /// next stream on this same encoder (same format and level, counters
    /// reset). The output is byte-identical to [`Encoder::finish`] on a fresh
    /// encoder; what carries over is memory: the sample and analysis buffers
    /// keep their capacity, and the LPC windows of a short final block are
    /// kept, so a caller that closes a stream per chunk stops rebuilding them
    /// (`libm::cos` per taper sample — a soft-float call on chips without an
    /// f64 FPU) and stops re-allocating its scratch on every chunk. The price
    /// is that those buffers stay allocated between chunks.
    pub fn finish_and_reset(&mut self) -> Vec<u8> {
        let out = self.encode_stream();
        for chan in &mut self.chans {
            chan.clear();
        }
        self.stats = EncodeStats::default();
        out
    }

    /// MD5 of the unencoded audio: interleaved samples, little-endian, at the
    /// coded bit depth — FLAC's STREAMINFO integrity signature.
    fn compute_md5(&self) -> [u8; 16] {
        let bytes_per = (self.bps / 8) as usize;
        let n = self.chans.first().map_or(0, |c| c.len());
        let ch = self.channels;
        let mut md5 = crate::md5::Md5::new();
        // Batch: build interleaved LE rows for a run of frames, hash per chunk.
        const CHUNK_FRAMES: usize = 16 * 1024;
        // Reserve for the first (often only) chunk, not the full CHUNK_FRAMES:
        // a short block used a 32 KiB transient here where 1 KiB sufficed —
        // material on a small no_std heap. resize() still grows it for a long
        // stream, so the per-chunk hashing is unchanged.
        let mut buf: Vec<u8> = Vec::with_capacity(CHUNK_FRAMES.min(n) * ch * bytes_per);
        let mut i = 0usize;
        while i < n {
            let end = (i + CHUNK_FRAMES).min(n);
            buf.resize((end - i) * ch * bytes_per, 0);
            match (bytes_per, ch) {
                // The hot shapes fill a preallocated chunk (no per-sample
                // Vec bookkeeping); the rest go generic.
                (2, 1) => {
                    let a = &self.chans[0][i..end];
                    for (out, &v) in buf.chunks_exact_mut(2).zip(a) {
                        out.copy_from_slice(&(v as i16).to_le_bytes());
                    }
                }
                (2, 2) => {
                    // One packed little-endian store per stereo frame (two
                    // 2-byte copies before), which the compiler vectorizes.
                    let (l, r) = (&self.chans[0][i..end], &self.chans[1][i..end]);
                    for ((out, &a), &b) in buf.chunks_exact_mut(4).zip(l).zip(r) {
                        let w = (a as u16 as u32) | ((b as u16 as u32) << 16);
                        out.copy_from_slice(&w.to_le_bytes());
                    }
                }
                (3, 2) => {
                    // Both 24-bit samples of a frame packed into one 48-bit
                    // value and written as six bytes in one copy.
                    let (l, r) = (&self.chans[0][i..end], &self.chans[1][i..end]);
                    for ((out, &a), &b) in buf.chunks_exact_mut(6).zip(l).zip(r) {
                        let w =
                            (a as u32 & 0xff_ffff) as u64 | ((b as u32 & 0xff_ffff) as u64) << 24;
                        out.copy_from_slice(&w.to_le_bytes()[..6]);
                    }
                }
                _ => {
                    for (j, out) in buf.chunks_exact_mut(ch * bytes_per).enumerate() {
                        for c in 0..ch {
                            out[c * bytes_per..(c + 1) * bytes_per]
                                .copy_from_slice(&self.chans[c][i + j].to_le_bytes()[..bytes_per]);
                        }
                    }
                }
            }
            md5.update(&buf);
            i = end;
        }
        md5.finalize()
    }

    fn encode_stream(&mut self) -> Vec<u8> {
        // RUSTY_FLAC_TIMING=1: print coarse stage shares to stderr (wiring
        // audit / campaign tool; zero cost when unset).
        #[cfg(feature = "std")]
        let timing = std::env::var_os("RUSTY_FLAC_TIMING").is_some();
        #[cfg(feature = "std")]
        let t0 = std::time::Instant::now();
        let n = self.chans.first().map_or(0, |c| c.len());
        let bps = self.bps;

        // Whole-stream output estimate: raw size is the ceiling for lossless.
        let raw = n * self.channels * (bps as usize / 8);
        let mut frames: Vec<u8> = Vec::with_capacity(raw / 2 + 4096);
        let (mut min_fs, mut max_fs) = (u32::MAX, 0u32);
        let mut frame_number = 0u64;
        let mut start = 0usize;
        let mut wins = core::mem::take(&mut self.wins);
        wins.wide = bps > 24;
        // One frame writer, reused (cleared) for every frame instead of a
        // fresh allocation per frame. Sized to the raw ceiling of the largest
        // (first) block — blocks are non-increasing, and a lossless frame
        // never exceeds raw — so it never reallocates.
        let first_bs = n.min(self.block_size);
        let mut frame_bw =
            BitWriter::with_capacity(first_bs * self.channels * (bps as usize / 8) + 64);
        while start < n {
            let bs = (n - start).min(self.block_size);
            wins.ensure(bs, self.block_size);
            self.encode_frame(frame_number, start, bs, bps, &wins, &mut frame_bw);
            let flen = frame_bw.bytes().len() as u32;
            min_fs = min_fs.min(flen);
            max_fs = max_fs.max(flen);
            frames.extend_from_slice(frame_bw.bytes());
            start += bs;
            frame_number += 1;
            self.stats.frames += 1;
        }
        self.wins = wins;
        if frames.is_empty() {
            min_fs = 0;
            max_fs = 0;
        }

        #[cfg(feature = "std")]
        let t_frames = t0.elapsed();
        #[cfg(feature = "std")]
        let t1 = std::time::Instant::now();

        // STREAMINFO (34 bytes). Block sizes are the NOMINAL blocking (the
        // spec's min/max exclude the final short block, and requires >= 16 —
        // a 5-sample stream still declares its nominal block size, like libFLAC).
        let mut si = BitWriter::with_capacity(34);
        si.write_bits(self.block_size as u64, 16);
        si.write_bits(self.block_size as u64, 16);
        si.write_bits(min_fs as u64, 24);
        si.write_bits(max_fs as u64, 24);
        si.write_bits(self.sample_rate as u64, 20);
        si.write_bits((self.channels as u64) - 1, 3);
        si.write_bits((bps as u64) - 1, 5);
        si.write_bits(n as u64, 36);
        for &byte in &self.compute_md5() {
            si.write_bits(byte as u64, 8);
        }
        let si = si.into_bytes();
        #[cfg(feature = "std")]
        if timing {
            eprintln!(
                "rusty_flac timing: frames {:.1} ms, md5+streaminfo {:.1} ms",
                t_frames.as_secs_f64() * 1e3,
                t1.elapsed().as_secs_f64() * 1e3
            );
        }

        let mut stream = Vec::with_capacity(4 + 4 + si.len() + frames.len());
        stream.extend_from_slice(b"fLaC");
        // Metadata block header: last-block=1, type=0 (STREAMINFO), length=34.
        stream.push(0x80);
        let len = si.len() as u32;
        stream.extend_from_slice(&[(len >> 16) as u8, (len >> 8) as u8, len as u8]);
        stream.extend_from_slice(&si);
        stream.extend_from_slice(&frames);
        stream
    }

    fn encode_frame(
        &mut self,
        frame_number: u64,
        start: usize,
        bs: usize,
        bps: u32,
        wins: &WindowCache,
        bw: &mut BitWriter,
    ) {
        // Decide the channel layout: stereo picks the cheapest decorrelation
        // mode; mono / multichannel code each channel independently.
        //
        // Every subframe borrows its samples straight out of `self.chans`
        // instead of copying them: the channel buffers are taken into
        // `held_chans` for the duration of the frame (so the borrows are of a
        // local, disjoint from `self.stats`/`self.scratch`) and restored after
        // the frame is written. Stereo's mid and side are computed, so those
        // subframes own their samples (`Cow::Owned`, moved, not copied).
        let held_chans: Vec<Vec<i32>> = core::mem::take(&mut self.chans);
        let (assignment, subframes): (u64, Vec<Subframe<'_>>) = if self.channels == 2 && bps <= 24 {
            // Borrowed from the held channels, so a chosen left or right arm
            // is handed back as a borrow rather than a copy.
            let (assignment, subs) = decide_stereo(
                &held_chans[0][start..start + bs],
                &held_chans[1][start..start + bs],
                bps,
                self.max_lpc_order,
                wins,
                &mut self.stats,
                &mut self.scratch,
            );
            match assignment {
                1 => self.stats.stereo_independent += 1,
                8 => self.stats.stereo_left_side += 1,
                9 => self.stats.stereo_right_side += 1,
                _ => self.stats.stereo_mid_side += 1,
            }
            (assignment, subs)
        } else {
            let max_lpc_order = self.max_lpc_order;
            let seg_windows = wide_seg_windows(max_lpc_order);
            let chans = &held_chans;
            let stats = &mut self.stats;
            let scratch = &mut self.scratch;
            let subs = (0..chans.len())
                .map(|c| {
                    let arm = ArmInput::prepare(&chans[c][start..start + bs], bps);
                    let choice = if bps > 24 {
                        analyze_wide(&arm, max_lpc_order, seg_windows, wins, stats, scratch)
                    } else {
                        analyze_subframe(&arm, max_lpc_order, wins, stats, scratch)
                    };
                    let ebps = arm.ebps;
                    (arm.into_cow(), ebps, choice)
                })
                .collect();
            ((chans.len() as u64) - 1, subs)
        };

        // The frame writer is owned by the caller and cleared here, so a whole
        // stream reuses one buffer instead of allocating one per frame.
        bw.clear();
        // --- frame header ---
        bw.write_bits(0x3FFE, 14); // sync
        bw.write_bits(0, 1); // reserved (mandatory 0)
        bw.write_bits(0, 1); // blocking strategy: fixed block size
        bw.write_bits(7, 4); // block-size code 7 => explicit 16-bit (bs-1) below
        bw.write_bits(0, 4); // sample-rate code 0 => from STREAMINFO
        bw.write_bits(assignment, 4); // 0/1..7 = independent, 8/9/10 = L-S / R-S / M-S
        bw.write_bits(sample_size_code(bps), 3);
        bw.write_bits(0, 1); // reserved (mandatory 0)
        write_utf8(&mut *bw, frame_number);
        bw.write_bits((bs as u64) - 1, 16); // block size - 1
        let hcrc = crc8(bw.bytes());
        bw.write_bits(hcrc as u64, 8);

        // --- subframes (each at its own bit depth; side channels use bps+1) ---
        for (samples, sf_bps, choice) in &subframes {
            write_subframe_from(&mut *bw, samples, *sf_bps, choice, &mut self.stats);
        }

        // --- frame footer: pad to byte, then CRC-16 of the whole frame ---
        bw.align_to_byte();
        let fcrc = crc16(bw.bytes());
        bw.write_bits(fcrc as u64, 16);
        // The subframes (and their borrow of held_chans) are written; restore
        // the channel buffers taken for the frame. The frame bytes stay in
        // `bw` for the caller to copy out.
        self.chans = held_chans;
    }
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

/// The two apodization windows tried per LPC candidate (Tukey, `alpha` 0.5
/// and 0.2).
const WINDOW_ALPHAS: [f64; 2] = [0.5, 0.2];

/// Window values are Q15 integers: `w` in `[0, 1]` is `round(w · 2^15)`, so
/// the flat middle (exactly 1.0) is `1 << 15` and windowing is an integer
/// multiply — the autocorrelation that follows is exact integer arithmetic.
const WIN_Q: u32 = 15;

/// The wide path's window scale. Q15 rounding error multiplies the signal by
/// noise near 2^-16 in the tapers, a broadband floor about 100 dB under the
/// signal; oversampled content (an empty band above the original
/// bandwidth) needs a predictor that sees far deeper than that. Q30: a
/// 31-bit sample times a Q30 window is below 2^61, still exact in i64.
const WIDE_WIN_Q: u32 = 30;

/// A Tukey window held as its two cosine tapers (Q15): `head` covers samples
/// `0..head.len()`, `tail` the last `tail.len()`, and every sample between
/// them is exactly `1 << WIN_Q` — so it is never stored.
#[derive(Clone, Copy)]
struct Window<'a, T = u16> {
    head: &'a [T],
    tail: &'a [T],
}

/// A window value's integer type and its fixed-point scale: Q15 `u16` on the
/// narrow path, Q30 `u32` on the wide one.
trait TaperWord: Copy {
    /// Fractional bits: the flat middle is `1 << Q`.
    const Q: u32;
    fn widen(self) -> i64;
}

impl TaperWord for u16 {
    const Q: u32 = WIN_Q;
    #[inline(always)]
    fn widen(self) -> i64 {
        self as i64
    }
}

impl TaperWord for u32 {
    const Q: u32 = WIDE_WIN_Q;
    #[inline(always)]
    fn widen(self) -> i64 {
        self as i64
    }
}

/// Owned tapers of one window, for a block size with no static table.
struct Tapers<T = u16> {
    head: Vec<T>,
    tail: Vec<T>,
}

impl<T> Default for Tapers<T> {
    fn default() -> Self {
        Tapers {
            head: Vec::new(),
            tail: Vec::new(),
        }
    }
}

impl<T> Tapers<T> {
    fn view(&self) -> Window<'_, T> {
        Window {
            head: &self.head,
            tail: &self.tail,
        }
    }
}

/// Segments of a block the wide path's extra windows are built from:
/// `(d, i, j)` is samples `n·i/d .. n·j/d`, each under its own Tukey window
/// ([`PIECE_ALPHA`]). A window that is zero outside a segment has the
/// autocorrelation of that segment alone, so partial-Tukey windows need no new
/// window type; and for lags shorter than a gap between two segments no
/// product straddles it, so a punchout window's autocorrelation is the sum of
/// its pieces'.
const PIECES: [(usize, usize, usize); 7] = [
    (2, 0, 1),
    (2, 1, 2),
    (3, 0, 1),
    (3, 1, 2),
    (3, 2, 3),
    (3, 1, 3),
    (3, 0, 2),
];

/// The wide path's extra windows, as the [`PIECES`] each sums: partial
/// Tukey (2) and (3) (libFLAC's `partial_tukey`), then punchout Tukey (3)
/// (the block with one third cut out).
const SEG_WINDOWS: [&[usize]; 8] = [&[0], &[1], &[2], &[3], &[4], &[5], &[2, 4], &[6]];

/// Taper fraction of each piece's own Tukey window: fully tapered (Hann).
/// On a 384 kHz / 32-bit track, 0.5 gave +0.6% and 0.25 +1.6% larger output.
const PIECE_ALPHA: f64 = 1.0;

/// Smallest block the extra windows are tried on (a third of it is the
/// shortest piece, which must hold far more samples than the LPC order).
const SEG_MIN_BLOCK: usize = 256;

/// The wide path's windows for one block size, all in Q30
/// ([`WIDE_WIN_Q`]): the two full-block Tukey windows ([`WINDOW_ALPHAS`])
/// and, for blocks of at least [`SEG_MIN_BLOCK`], the [`PIECES`]. Built at
/// run time (once per block size), so unlike the narrow path's static
/// full-block table they follow the build's `cos`; `--features libm` makes
/// them identical everywhere.
#[derive(Default)]
struct WideWindows {
    /// Block size this set is for (0: none).
    n: usize,
    full: [Tapers<u32>; 2],
    /// Empty below [`SEG_MIN_BLOCK`].
    pieces: Vec<Tapers<u32>>,
}

impl WideWindows {
    fn build(&mut self, n: usize) {
        self.n = n;
        for (t, &alpha) in self.full.iter_mut().zip(&WINDOW_ALPHAS) {
            *t = tukey_tapers_q(n, alpha, q30);
        }
        self.pieces.clear();
        if n >= SEG_MIN_BLOCK {
            for &(d, i, j) in &PIECES {
                let len = n * j / d - n * i / d;
                self.pieces.push(tukey_tapers_q(len, PIECE_ALPHA, q30));
            }
        }
    }
}

/// The windows for the current block size. A [`BLOCK_SIZE`] block reads a
/// static table; the nominal block size set by [`Encoder::set_block_size`]
/// and a stream's short final block each build and keep their own tapers.
#[derive(Default)]
struct WindowCache {
    /// Block size the windows are currently for.
    n: usize,
    /// `n` is the stream's nominal block size (every block but a short last).
    full: bool,
    /// Block size `nominal` was built for (0: none): a nominal block size
    /// other than [`BLOCK_SIZE`], which has no static table.
    nominal_n: usize,
    nominal: [Tapers; 2],
    /// Block size `short` was built for (0: none). Kept apart from `n`, so a
    /// stream's full blocks do not evict its short tail's tapers — the next
    /// stream of the same length reuses them.
    short_n: usize,
    short: [Tapers; 2],
    /// Levels 0-2: analyse with the first window only (see
    /// `set_compression_level`). False -- both windows -- by default.
    single: bool,
    /// Wide path only (32-bit streams): the [`WideWindows`] for full blocks
    /// and for the short final block.
    wide: bool,
    wide_full: WideWindows,
    wide_short: WideWindows,
    /// Taper builds, for the reuse test: a cache that misses is otherwise
    /// invisible, the output being byte-identical either way.
    #[cfg(test)]
    builds: usize,
}

impl WindowCache {
    /// Make the windows for an `n`-sample block of a stream framed at
    /// `nominal` samples.
    fn ensure(&mut self, n: usize, nominal: usize) {
        self.n = n;
        self.full = n == nominal;
        let (built_n, tapers) = if self.full {
            (&mut self.nominal_n, &mut self.nominal)
        } else {
            (&mut self.short_n, &mut self.short)
        };
        if n != BLOCK_SIZE && *built_n != n {
            *built_n = n;
            #[cfg(test)]
            {
                self.builds += 1;
            }
            for (slot, &alpha) in tapers.iter_mut().zip(&WINDOW_ALPHAS) {
                *slot = tukey_tapers(n, alpha);
            }
        }
        if self.wide {
            let set = if self.full {
                &mut self.wide_full
            } else {
                &mut self.wide_short
            };
            if set.n != n {
                set.build(n);
            }
        }
    }

    /// The wide path's windows for the current block.
    fn wide(&self) -> &WideWindows {
        let set = if self.full {
            &self.wide_full
        } else {
            &self.wide_short
        };
        debug_assert!(self.wide && set.n == self.n, "wide windows not built");
        set
    }

    /// Windows the analysis tries: one at levels 0-2, both above.
    fn count(&self) -> usize {
        if self.single {
            1
        } else {
            WINDOW_ALPHAS.len()
        }
    }

    fn get(&self, k: usize) -> Window<'_> {
        if self.n == BLOCK_SIZE {
            full_block_window(k)
        } else if self.full {
            self.nominal[k].view()
        } else {
            self.short[k].view()
        }
    }
}

/// The [`BLOCK_SIZE`] windows: a generated table, identical to
/// [`tukey_tapers`] (gated by `window_table_is_runtime_tukey`). Integer, so
/// one table serves `libm` and platform-libm builds alike.
fn full_block_window(k: usize) -> Window<'static> {
    use crate::window_table::{HEAD_0, HEAD_1, TAIL_0, TAIL_1};
    if k == 0 {
        Window {
            head: &HEAD_0,
            tail: &TAIL_0,
        }
    } else {
        Window {
            head: &HEAD_1,
            tail: &TAIL_1,
        }
    }
}

/// A window value in `[0, 1]` as Q15, rounded half up (`w · 2^15` is exact
/// and far below 2^52, so `+ 0.5` then truncation is exact rounding).
fn q15(w: f64) -> u16 {
    (w * (1u32 << WIN_Q) as f64 + 0.5) as u16
}

/// A Tukey window of `n` samples as its two cosine tapers in Q15: the
/// full-length window's per-sample arithmetic (kept below as the test oracle),
/// evaluated only where the window is not `1.0`. `x = i/(n-1)` is monotonic
/// in `i`, so the head is a prefix and the tail a suffix.
fn tukey_tapers(n: usize, alpha: f64) -> Tapers {
    tukey_tapers_q(n, alpha, q15)
}

/// A window value in `[0, 1]` as Q30, rounded half up (exact as for [`q15`]).
fn q30(w: f64) -> u32 {
    (w * (1u32 << WIDE_WIN_Q) as f64 + 0.5) as u32
}

/// [`tukey_tapers`] at any fixed-point scale `quant`.
fn tukey_tapers_q<T>(n: usize, alpha: f64, quant: fn(f64) -> T) -> Tapers<T> {
    let mut t = Tapers::default();
    if n <= 1 {
        return t;
    }
    let x_of = |i: usize| i as f64 / (n - 1) as f64;
    let mut i = 0;
    while i < n && x_of(i) < alpha / 2.0 {
        let x = x_of(i);
        t.head.push(quant(
            0.5 * (1.0 + math::cos(core::f64::consts::PI * (2.0 * x / alpha - 1.0))),
        ));
        i += 1;
    }
    let mut j = n;
    while j > i && x_of(j - 1) > 1.0 - alpha / 2.0 {
        j -= 1;
    }
    for idx in j..n {
        let x = x_of(idx);
        t.tail.push(quant(
            0.5 * (1.0 + math::cos(core::f64::consts::PI * (2.0 * x / alpha - 2.0 / alpha + 1.0))),
        ));
    }
    t
}

/// Tukey apodization window: flat middle with cosine tapers. The original
/// full-length float form, kept as the oracle for [`tukey_tapers`].
#[cfg(test)]
fn tukey_window(n: usize, alpha: f64) -> Vec<f64> {
    let mut w = vec![1.0f64; n];
    if n <= 1 {
        return w;
    }
    for (i, wi) in w.iter_mut().enumerate() {
        let x = i as f64 / (n - 1) as f64;
        if x < alpha / 2.0 {
            *wi = 0.5 * (1.0 + math::cos(core::f64::consts::PI * (2.0 * x / alpha - 1.0)));
        } else if x > 1.0 - alpha / 2.0 {
            *wi = 0.5
                * (1.0 + math::cos(core::f64::consts::PI * (2.0 * x / alpha - 2.0 / alpha + 1.0)));
        }
    }
    w
}

// ---------------------------------------------------------------------------
// Frame-header helpers
// ---------------------------------------------------------------------------

/// FLAC's UTF-8-style coding of the frame number (fixed blocking strategy).
fn write_utf8(bw: &mut BitWriter, val: u64) {
    if val < 0x80 {
        bw.write_bits(val, 8);
        return;
    }
    let nconts: u32 = if val < 0x800 {
        1
    } else if val < 0x1_0000 {
        2
    } else if val < 0x20_0000 {
        3
    } else if val < 0x400_0000 {
        4
    } else {
        5
    };
    let lead_ones = nconts + 1;
    let prefix = (((1u64 << lead_ones) - 1) << (8 - lead_ones)) & 0xFF;
    bw.write_bits(prefix | (val >> (6 * nconts)), 8);
    for i in (0..nconts).rev() {
        bw.write_bits(0x80 | ((val >> (6 * i)) & 0x3F), 8);
    }
}

/// FLAC frame-header sample-size code for a bit depth.
fn sample_size_code(bps: u32) -> u64 {
    match bps {
        8 => 1,
        12 => 2,
        16 => 4,
        20 => 5,
        24 => 6,
        32 => 7,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Residual coding — exact Rice costs via per-partition shifted sums
// ---------------------------------------------------------------------------

/// Zigzag-fold a signed residual to the unsigned value FLAC Rice-codes.
#[inline]
fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

/// `sums[k] = Σ (zigzag(v) >> k)` over a residual slice, for k = 0..=14.
/// The exact Rice bit cost at parameter k is `sums[k] + cnt·(1 + k)` — one
/// pass yields every parameter's exact cost. Integer sums, so the AVX2 path
/// is exact (gated by `rice_sums_avx2_matches_scalar`).
///
/// Only `out.len()` sums are produced (k = 0..out.len()): the caller sizes
/// `out` to the residual's top bit, above which every sum is zero.
#[inline]
fn rice_sums_into(res: &[i32], out: &mut [u64]) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // A row of at most 16 sums means every zigzagged value is below
            // 2^16 (`rice_stride`), so a partition of fewer than 2^16 of them
            // sums inside u32: eight lanes a vector, two vectors for k 0..15.
            if out.len() <= 16 && res.len() < 1 << 16 {
                // SAFETY: guarded by the runtime AVX2 check.
                unsafe { rice_sums_avx2_u32(res, out) };
                return;
            }
            // SAFETY: guarded by the runtime AVX2 check.
            let all = unsafe { rice_sums_avx2(res) };
            out.copy_from_slice(&all[..out.len()]);
            return;
        }
    }
    rice_sums_scalar_into(res, out);
}

/// Scalar shifted sums for k = 0..out.len() — on a chip without SIMD this is
/// one u64 add per sample per k, so stopping at the top bit is the saving.
///
/// `out` is sized by [`rice_stride`]: when it stops short of `RICE_KMAX + 1`,
/// every zigzagged value is below `2^(out.len() - 1)`. If `res.len()` of those
/// also fit 32 bits — always, for 16-bit audio — the sums are accumulated in
/// u32 (exact, and one add per term instead of a 64-bit add with carry on a
/// 32-bit core).
fn rice_sums_scalar_into(res: &[i32], out: &mut [u64]) {
    let top = out.len() - 1;
    let cnt_bits = usize::BITS - res.len().leading_zeros();
    if out.len() <= RICE_KMAX && top as u32 + cnt_bits <= 32 {
        debug_assert!(
            res.iter().all(|&v| zigzag(v) >> top == 0),
            "row below top bit"
        );
        // Rows of up to 20 sums (all of 16-bit audio) use a body unrolled at
        // that width: every shift an immediate, every accumulator a register
        // (constant indices), no per-k loop. The generic loop below costs 9
        // instructions per (sample, k) on Xtensa (shift-amount setup, load,
        // add, store, loop control); opt-level "s" does not unroll it.
        if let Some(row) = RICE_ROWS.get(out.len()) {
            return row(res, out);
        }
        let mut acc = [0u32; RICE_KMAX];
        let acc = &mut acc[..out.len()];
        for &v in res {
            let u = zigzag(v);
            for (k, a) in acc.iter_mut().enumerate() {
                *a += u >> k;
            }
        }
        for (o, &a) in out.iter_mut().zip(acc.iter()) {
            *o = a as u64;
        }
        return;
    }
    out.fill(0);
    for &v in res {
        let u = zigzag(v);
        for (k, s) in out.iter_mut().enumerate() {
            *s += (u >> k) as u64;
        }
    }
}

/// Unrolled u32 row bodies for [`rice_sums_scalar_into`], one per width.
macro_rules! rice_rows {
    ($($name:ident, $n:literal, $($k:literal)+;)+) => {
        $(
            fn $name(res: &[i32], out: &mut [u64]) {
                let mut acc = [0u32; $n];
                for &v in res {
                    let u = zigzag(v);
                    $(acc[$k] += u >> $k;)+
                }
                for (o, a) in out.iter_mut().zip(acc) {
                    *o = a as u64;
                }
            }
        )+
    };
}
rice_rows! {
    rice_row_1, 1, 0;
    rice_row_2, 2, 0 1;
    rice_row_3, 3, 0 1 2;
    rice_row_4, 4, 0 1 2 3;
    rice_row_5, 5, 0 1 2 3 4;
    rice_row_6, 6, 0 1 2 3 4 5;
    rice_row_7, 7, 0 1 2 3 4 5 6;
    rice_row_8, 8, 0 1 2 3 4 5 6 7;
    rice_row_9, 9, 0 1 2 3 4 5 6 7 8;
    rice_row_10, 10, 0 1 2 3 4 5 6 7 8 9;
    rice_row_11, 11, 0 1 2 3 4 5 6 7 8 9 10;
    rice_row_12, 12, 0 1 2 3 4 5 6 7 8 9 10 11;
    rice_row_13, 13, 0 1 2 3 4 5 6 7 8 9 10 11 12;
    rice_row_14, 14, 0 1 2 3 4 5 6 7 8 9 10 11 12 13;
    rice_row_15, 15, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14;
    rice_row_16, 16, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15;
    rice_row_17, 17, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16;
    rice_row_18, 18, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17;
    rice_row_19, 19, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18;
    rice_row_20, 20, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19;
}

/// Row bodies by width (index 0 unused: a row always has k = 0).
type RiceRow = fn(&[i32], &mut [u64]);
static RICE_ROWS: [RiceRow; 21] = [
    rice_row_1,
    rice_row_1,
    rice_row_2,
    rice_row_3,
    rice_row_4,
    rice_row_5,
    rice_row_6,
    rice_row_7,
    rice_row_8,
    rice_row_9,
    rice_row_10,
    rice_row_11,
    rice_row_12,
    rice_row_13,
    rice_row_14,
    rice_row_15,
    rice_row_16,
    rice_row_17,
    rice_row_18,
    rice_row_19,
    rice_row_20,
];

/// Number of shifted sums worth keeping for a residual: k = 0..=top, where
/// `top` is the bit length of the largest zigzagged value (capped at
/// RICE_KMAX). Every sum at k >= that bit length is zero, and
/// `best_k_from_sums`' convex scan stops at the first k whose cost rises —
/// which it does at the first all-zero k (cost `cnt·(1+k)` grows by `cnt`) —
/// so a scan over the truncated row chooses exactly what a full one would.
fn rice_stride(res: &[i32]) -> usize {
    let or = res.iter().fold(0u32, |a, &v| a | zigzag(v));
    ((32 - or.leading_zeros()) as usize).min(RICE_KMAX) + 1
}

/// [`rice_sums_avx2`] in u32 lanes, for rows of at most 16 sums: eight k per
/// vector, so two shift-add pairs a sample where the u64 kernel needs four,
/// and the row is written straight into `out` (no 31-entry stack row, no
/// copy). The caller guarantees every value is below 2^16 and there are fewer
/// than 2^16 of them, so no lane can overflow.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx2")]
unsafe fn rice_sums_avx2_u32(res: &[i32], out: &mut [u64]) {
    use core::arch::x86_64::*;
    debug_assert!(out.len() <= 16 && res.iter().all(|&v| zigzag(v) < 1 << 16));
    let sh0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let sh1 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    let mut a0 = _mm256_setzero_si256();
    let mut a1 = _mm256_setzero_si256();
    for &v in res {
        let b = _mm256_set1_epi32(zigzag(v) as i32);
        a0 = _mm256_add_epi32(a0, _mm256_srlv_epi32(b, sh0));
        a1 = _mm256_add_epi32(a1, _mm256_srlv_epi32(b, sh1));
    }
    let mut lanes = [0u32; 16];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, a0);
    _mm256_storeu_si256(lanes.as_mut_ptr().add(8) as *mut __m256i, a1);
    for (o, &l) in out.iter_mut().zip(&lanes) {
        *o = l as u64;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx2")]
unsafe fn rice_sums_avx2(res: &[i32]) -> [u64; RICE_KMAX + 1] {
    use core::arch::x86_64::*;
    // 16 u64 accumulator lanes for k 0..15; k 16..=30 is folded from lane 15
    // afterwards by re-scanning IF any residual is big enough to need it
    // (rare outside high-entropy 24-bit content), so the common path stays 4
    // shift/add register pairs per sample.
    let sh0 = _mm256_setr_epi64x(0, 1, 2, 3);
    let sh1 = _mm256_setr_epi64x(4, 5, 6, 7);
    let sh2 = _mm256_setr_epi64x(8, 9, 10, 11);
    let sh3 = _mm256_setr_epi64x(12, 13, 14, 15);
    let mut a0 = _mm256_setzero_si256();
    let mut a1 = _mm256_setzero_si256();
    let mut a2 = _mm256_setzero_si256();
    let mut a3 = _mm256_setzero_si256();
    let mut or_acc = 0u32;
    for &v in res {
        let u = zigzag(v);
        or_acc |= u;
        let b = _mm256_set1_epi64x(u as i64);
        a0 = _mm256_add_epi64(a0, _mm256_srlv_epi64(b, sh0));
        a1 = _mm256_add_epi64(a1, _mm256_srlv_epi64(b, sh1));
        a2 = _mm256_add_epi64(a2, _mm256_srlv_epi64(b, sh2));
        a3 = _mm256_add_epi64(a3, _mm256_srlv_epi64(b, sh3));
    }
    let mut lanes = [0u64; 16];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, a0);
    _mm256_storeu_si256(lanes.as_mut_ptr().add(4) as *mut __m256i, a1);
    _mm256_storeu_si256(lanes.as_mut_ptr().add(8) as *mut __m256i, a2);
    _mm256_storeu_si256(lanes.as_mut_ptr().add(12) as *mut __m256i, a3);
    let mut sums = [0u64; RICE_KMAX + 1];
    sums[..16].copy_from_slice(&lanes);
    // High-k tail: only when some residual exceeds 16 bits after folding, and
    // only up to the top set bit (sums above it are zero by construction).
    if or_acc >> 16 != 0 {
        let top = (32 - or_acc.leading_zeros() as usize).min(RICE_KMAX);
        for &v in res {
            let u = zigzag(v);
            for (k, s) in sums.iter_mut().enumerate().take(top + 1).skip(16) {
                *s += (u >> k) as u64;
            }
        }
    }
    sums
}

/// Best Rice parameter within `0..=kmax` (lowest k on ties) and its exact
/// body bit cost, from precomputed shifted sums. The cost is convex in k
/// (unary halves, suffix grows by cnt), so the scan stops at the first rise.
///
/// The scan starts at the bit length of the mean value, not at k = 0, and
/// walks downhill: down while the step below costs no more (so ties settle
/// on the lowest k), otherwise up while the step above costs less. On a
/// convex cost that is the same answer as the ascending scan
/// (`best_k_scan`, the test oracle), in two or three evaluations where the
/// ascending scan took k + 1 -- the planner runs it on every partition of
/// every level, twice.
#[inline]
fn best_k_from_sums(sums: &[u64], cnt: u64, kmax: usize) -> (u32, u64) {
    let top = kmax.min(sums.len() - 1);
    let cost = |k: usize| sums[k] + cnt * (1 + k as u64);
    let mean = sums[0].checked_div(cnt).unwrap_or(0);
    let mut k = ((u64::BITS - mean.leading_zeros()) as usize)
        .saturating_sub(1)
        .min(top);
    let mut c = cost(k);
    if k > 0 && cost(k - 1) <= c {
        while k > 0 {
            let d = cost(k - 1);
            if d > c {
                break;
            }
            k -= 1;
            c = d;
        }
    } else {
        while k < top {
            let d = cost(k + 1);
            if d >= c {
                break;
            }
            k += 1;
            c = d;
        }
    }
    (k as u32, c)
}

/// The ascending scan [`best_k_from_sums`] replaced: first rise ends it.
#[cfg(test)]
fn best_k_scan(sums: &[u64], cnt: u64, kmax: usize) -> (u32, u64) {
    let mut best_k = 0u32;
    let mut best = sums[0] + cnt;
    for (k, &s) in sums.iter().enumerate().take(kmax + 1).skip(1) {
        let b = s + cnt * (1 + k as u64);
        if b < best {
            best = b;
            best_k = k as u32;
        } else {
            break; // convex: it only grows from here
        }
    }
    (best_k, best)
}

/// A residual coding plan: the coding method (0 = 4-bit Rice params, 1 =
/// Rice2 5-bit params), the chosen partition order, per-partition parameters,
/// and the residual-body bit cost (Σ param-field + Rice codes).
struct ResidualPlan {
    method: u32,
    partition_order: u32,
    ks: Vec<u32>,
    bits: u64,
}

/// Highest Rice partition order searched (128 partitions of a 4096 block).
///
/// It was 8 (256 partitions of 16 samples): the planner paid for 256 finest
/// partitions on every candidate, each a shifted-sum kernel call on 16
/// samples where its fixed cost dominates. Order 8 changed no file of the
/// corpus. libFLAC's `-8` stops at 6, but 6 costs noise-like content
/// (applause grew 0.40%) and 7 recovers it exactly.
const MAX_PARTITION_ORDER: u32 = 7;

/// Largest usable partition order for a `bs`-sample block with predictor order
/// `p`. Capped at [`MAX_PARTITION_ORDER`].
fn max_partition_order(bs: usize, p: usize) -> u32 {
    let mut po = 0u32;
    while po < MAX_PARTITION_ORDER {
        let next = po + 1;
        if bs & ((1usize << next) - 1) != 0 {
            break; // bs not a multiple of 2^next
        }
        if (bs >> next) <= p {
            break; // partition 0 would be empty
        }
        po = next;
    }
    po
}

/// Choose the best partition order + per-partition Rice parameters, with costs
/// identical to an independent exhaustive scan per order (the original), but
/// computed in ONE pass: shifted sums per finest partition, merged pairwise
/// upward — O(15n) total instead of O(15n) per order.
fn plan_partitions(res: &[i32], bs: usize, p: usize, scratch: &mut EncodeScratch) -> ResidualPlan {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime AVX2 check.
            return unsafe { plan_partitions_avx2(res, bs, p, scratch) };
        }
    }
    plan_partitions_body(res, bs, p, scratch)
}

/// [`plan_partitions`] compiled for AVX2: its row-width pass (the zigzag OR
/// over the whole residual) ran at the SSE2 baseline, 15 instructions per 8
/// samples, and the finest-level kernels now inline into it. Same body.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx2")]
unsafe fn plan_partitions_avx2(
    res: &[i32],
    bs: usize,
    p: usize,
    scratch: &mut EncodeScratch,
) -> ResidualPlan {
    plan_partitions_body(res, bs, p, scratch)
}

#[inline(always)]
fn plan_partitions_body(
    res: &[i32],
    bs: usize,
    p: usize,
    scratch: &mut EncodeScratch,
) -> ResidualPlan {
    let max_po = max_partition_order(bs, p);
    let finest_parts = 1usize << max_po;
    let finest_size = bs >> max_po;

    // Partition-sum scratch and the two per-level Rice-parameter buffers are
    // all encoder-owned and reused across every plan for the encoder's life.
    // The sums were a fresh `Vec` per call on the no_std path (no
    // thread-local) — the largest single analysis allocation — and ks0/ks1
    // were a fresh pair per plan. Disjoint fields, so borrowed together.
    let EncodeScratch {
        words: sums,
        ks0,
        ks1,
        ..
    } = scratch;
    {
        let stride = rice_stride(res);
        regrow(sums, finest_parts * stride);
        sums.resize(finest_parts * stride, 0);
        let mut idx = 0usize;
        for (part, row) in sums.chunks_exact_mut(stride).enumerate() {
            let cnt = if part == 0 {
                finest_size - p
            } else {
                finest_size
            };
            rice_sums_into(&res[idx..idx + cnt], row);
            idx += cnt;
        }

        // Cost one level from `sums[..n_part]`, both methods (method 0 pays
        // 4 bits/param but caps k at 14; Rice2 pays 5 for k up to 30). The two
        // per-parameter buffers are reused across every level (cleared, not
        // reallocated) — a fresh pair per level was ~2 × (max_po + 1)
        // allocations per plan, the dominant no_std alloc count. Reserve up
        // front so the first fill does not grow them incrementally.
        ks0.clear();
        ks0.reserve(finest_parts);
        ks1.clear();
        ks1.reserve(finest_parts);

        // Evaluate from the finest level down, merging pairs in place. Taking
        // ties with `<=` while descending reproduces the ascending strict-<
        // search's lowest-po-wins-ties rule. The winning parameters are copied
        // into `best_ks` (one reused buffer) only when a level improves.
        let mut best_method = 0u32;
        let mut best_po = 0u32;
        let mut best_ks: Vec<u32> = Vec::new();
        let mut best_bits = u64::MAX;
        let mut po = max_po;
        loop {
            let n_part = 1usize << po;
            let psize = bs >> po;
            ks0.clear();
            ks1.clear();
            let (mut bits0, mut bits1) = (0u64, 0u64);
            for (part, s) in sums[..n_part * stride].chunks_exact(stride).enumerate() {
                let cnt = if part == 0 { psize - p } else { psize } as u64;
                let (k1, kb1) = best_k_from_sums(s, cnt, RICE_KMAX);
                let (k0, kb0) = if k1 as usize <= RICE_KMAX_M0 {
                    (k1, kb1)
                } else {
                    best_k_from_sums(s, cnt, RICE_KMAX_M0)
                };
                ks0.push(k0);
                ks1.push(k1);
                bits0 += 4 + kb0;
                bits1 += 5 + kb1;
            }
            let (method, bits) = if bits1 < bits0 {
                (1, bits1)
            } else {
                (0, bits0)
            };
            if bits <= best_bits {
                best_method = method;
                best_po = po;
                best_bits = bits;
                best_ks.clear();
                best_ks.extend_from_slice(if method == 1 { ks1 } else { ks0 });
            }
            if po == 0 {
                break;
            }
            // Merge pairs into the front half for the next-coarser level: row
            // i = row 2i + row 2i+1. Row i is written only after rows <= 2i+1
            // are read, so ascending order is safe in place.
            //
            // Through split slices and zipped rows, not `sums[..]` indexing:
            // three bounds checks per element kept the loop scalar at 15
            // instructions an element, a third of the planner. Row 0 is the
            // only one whose destination is one of its own sources, so it is
            // done in place first; every later row's destination lies below
            // both of its sources.
            {
                let (r0, rest) = sums.split_at_mut(stride);
                for (d, &b) in r0.iter_mut().zip(&rest[..stride]) {
                    *d += b;
                }
            }
            for i in 1..n_part / 2 {
                let (head, tail) = sums.split_at_mut(2 * i * stride);
                let (a, b) = tail[..2 * stride].split_at(stride);
                for ((d, &x), &y) in head[i * stride..(i + 1) * stride].iter_mut().zip(a).zip(b) {
                    *d = x + y;
                }
            }
            po -= 1;
        }
        ResidualPlan {
            method: best_method,
            partition_order: best_po,
            ks: best_ks,
            bits: best_bits,
        }
    }
}

/// Write a partitioned Rice residual body.
fn write_partitioned_residual(
    bw: &mut BitWriter,
    res: &[i32],
    bs: usize,
    p: usize,
    plan: &ResidualPlan,
) {
    let n_part = 1usize << plan.partition_order;
    let psize = bs >> plan.partition_order;
    let param_bits = 4 + plan.method;
    let mut idx = 0usize;
    for part in 0..n_part {
        let cnt = if part == 0 { psize - p } else { psize };
        let k = plan.ks[part];
        bw.write_bits(k as u64, param_bits);
        bw.write_rice_run(&res[idx..idx + cnt], k);
        idx += cnt;
    }
}

// ---------------------------------------------------------------------------
// LPC
// ---------------------------------------------------------------------------

/// Empty `buf` and make room for `len` words. When it has to grow, the old
/// allocation is freed first: its contents are dead, and a grow through
/// `realloc` would hold old and new at once — on a first-fit embedded heap
/// that is the encoder's peak.
fn regrow(buf: &mut Vec<u64>, len: usize) {
    buf.clear();
    if buf.capacity() < len {
        *buf = Vec::with_capacity(len);
    }
}

/// Autocorrelation of the windowed samples, lags 0..=max_order, in exact
/// integer arithmetic; the result is in the units of the float
/// autocorrelation it replaced (`Σ (s·w)(s'·w')`, kept as `autocorr_f64` in
/// the tests), so Levinson and the order-selection estimate read it as before.
///
/// The windowed samples `x = s·q` (Q15 window `q`) are normalised by a
/// right shift so that `|x| ≤ 2^b` with `2b + ⌈log2 n⌉ ≤ 62`: every product
/// is at most 2^(2b) and a lag sum of at most n of them cannot overflow i64.
/// Integer sums are exact in any order, so host, chip and the AVX2 kernel
/// agree by construction — and on a chip without an f64 FPU each
/// multiply-add is a 32×32→64 multiply and an add instead of two soft-float
/// calls (the stage was 57–78 % of an ESP32-S3 encode;
/// docs/plans/esp32-encoder-cost.md item 2).
fn autocorrelation<T: TaperWord>(
    samples: &[i32],
    max_order: usize,
    win: Window<'_, T>,
    scratch: &mut EncodeScratch,
) -> u32 {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime AVX2 check.
            return unsafe { autocorrelation_avx2(samples, max_order, win, scratch) };
        }
    }
    autocorrelation_body(samples, max_order, win, scratch)
}

/// [`autocorrelation`] compiled for AVX2: the windowing pass is three
/// element-wise loops (peak fold, flat-middle widen-and-shift, tapered
/// multiply) that the x86-64 baseline vectorizes with SSE2 only -- the middle
/// copy was 15 instructions per four samples, where AVX2 sign-extends four
/// lanes in one. Same body, same integers.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx2")]
unsafe fn autocorrelation_avx2<T: TaperWord>(
    samples: &[i32],
    max_order: usize,
    win: Window<'_, T>,
    scratch: &mut EncodeScratch,
) -> u32 {
    autocorrelation_body(samples, max_order, win, scratch)
}

#[inline(always)]
fn autocorrelation_body<T: TaperWord>(
    samples: &[i32],
    max_order: usize,
    win: Window<'_, T>,
    scratch: &mut EncodeScratch,
) -> u32 {
    // Windowed samples and the autocorrelation output are both encoder-owned
    // buffers, reused across every subframe analysis. Result in
    // `scratch.autoc`.
    let w = &mut scratch.words;
    let (h, t, n) = (win.head.len(), win.tail.len(), samples.len());
    regrow(w, n);
    // Normalisation shift, from the samples: |x| = |s|·q < 2^(bits(s) + Q),
    // so `x >> sh` is at most 2^b in magnitude. (Derived from the samples
    // rather than from a pass over the products, and fused into the one
    // windowing pass below.)
    let lg = usize::BITS - n.saturating_sub(1).leading_zeros(); // ⌈log2 n⌉
    let b = (62 - lg) / 2;
    let big = samples.iter().fold(0u32, |a, &s| a | s.unsigned_abs());
    let q_bits = T::Q;
    let sh = (u32::BITS - big.leading_zeros() + q_bits).saturating_sub(b);
    // |s| <= 2^31 (32-bit encode), q <= 2^30: every s·q fits i64.
    w.extend(
        samples[..h]
            .iter()
            .zip(win.head)
            .map(|(&s, &q)| ((s as i64 * q.widen()) >> sh) as u64),
    );
    // The flat middle, (s << Q) >> sh, as one shift of s.
    if sh <= q_bits {
        let up = q_bits - sh;
        w.extend(samples[h..n - t].iter().map(|&s| ((s as i64) << up) as u64));
    } else {
        let down = sh - q_bits;
        w.extend(
            samples[h..n - t]
                .iter()
                .map(|&s| ((s >> down) as i64) as u64),
        );
    }
    w.extend(
        samples[n - t..]
            .iter()
            .zip(win.tail)
            .map(|(&s, &q)| ((s as i64 * q.widen()) >> sh) as u64),
    );
    let mut sums = [0i64; 33];
    let sums = &mut sums[..=max_order];
    autocorr_int(w, sums);
    // x = s·w·2^(Q−sh), so Σ x·x' = Σ (s·w)(s'·w') · 2^(2Q−2sh).
    let scale = pow2(2 * sh as i32 - 2 * q_bits as i32);
    let autoc = &mut scratch.autoc;
    autoc.clear();
    autoc.extend(sums.iter().map(|&v| v as f64 * scale));
    big
}

/// `2^e` as an f64, exactly (normal range only).
fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// Lag sums `out[lag] = Σ x[i]·x[i+lag]` over normalised windowed samples
/// (i64 bit patterns whose values fit i32).
fn autocorr_int(w: &[u64], out: &mut [i64]) {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime AVX2 check.
            unsafe { autocorr_int_avx2(w, out) };
            return;
        }
    }
    autocorr_int_scalar(w, out);
}

/// Scalar lag sums: a sign-extended 32×32→64 multiply and a 64-bit add per
/// term (native `mull`/`mulsh` on Xtensa, no libcall).
///
/// Lags go in pairs: one load of `x[i]` serves both, and the `x[i + lag + 1]`
/// loaded for the second lag is carried into the next step as the first
/// lag's `x[i + 1 + lag]` — two loads per two multiply-adds instead of four.
/// Integer sums, so the pairing cannot change a result.
fn autocorr_int_scalar(w: &[u64], out: &mut [i64]) {
    let n = w.len();
    // Carried values stay i32 and widen only at the multiply: an i64 carried
    // across iterations loses its sign-extension and becomes a full 64×64
    // multiply (measured +1,491,164 instructions over the bench rows).
    let x = |v: u64| v as i32;
    let mul = |a: i32, b: i32| a as i64 * b as i64;
    let mut lag = 0;
    while lag + 2 <= out.len() {
        // Lag + 1 covers i in 0..m; lag covers one more, i = m.
        let m = n - lag - 1;
        let (mut a0, mut a1) = (0i64, 0i64);
        let mut cur = x(w[lag]);
        for (&xi, &next) in w[..m].iter().zip(&w[lag + 1..]) {
            let (xi, next) = (x(xi), x(next));
            a0 += mul(xi, cur);
            a1 += mul(xi, next);
            cur = next;
        }
        a0 += mul(x(w[m]), cur);
        out[lag] = a0;
        out[lag + 1] = a1;
        lag += 2;
    }
    if lag < out.len() {
        out[lag] = w[..n - lag]
            .iter()
            .zip(&w[lag..])
            .map(|(&a, &b)| mul(x(a), x(b)))
            .sum();
    }
}

/// AVX2 lag sums: `_mm256_mul_epi32` is exactly the sign-extended low-32-bit
/// product the scalar twin forms, four samples at a time. Lags go in groups
/// of four sharing each `y` load, one accumulator per lag, so the loop is
/// bound by loads rather than by an add chain. Integer, so neither the lane
/// nor the group order matters (gated by `autocorr_int_avx2_matches_scalar`).
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx2")]
unsafe fn autocorr_int_avx2(w: &[u64], out: &mut [i64]) {
    // Lags in groups of up to eight: each group loads `x[i]` once for all of
    // its lags, and eight accumulators plus the shared load fit the sixteen
    // ymm registers. Groups of four split level 5's nine lags 4+4+1, the last
    // group reloading every sample for one multiply-add; now 8+1, and level
    // 8's thirteen 8+5 instead of 4+4+4+1.
    let mut lag0 = 0;
    while lag0 < out.len() {
        let k = (out.len() - lag0).min(8);
        match k {
            1 => lag_group::<1>(w, lag0, out),
            2 => lag_group::<2>(w, lag0, out),
            3 => lag_group::<3>(w, lag0, out),
            4 => lag_group::<4>(w, lag0, out),
            5 => lag_group::<5>(w, lag0, out),
            6 => lag_group::<6>(w, lag0, out),
            7 => lag_group::<7>(w, lag0, out),
            _ => lag_group::<8>(w, lag0, out),
        }
        lag0 += k;
    }
}

/// Lags `lag0..lag0 + K` of [`autocorr_int_avx2`]: vector over the range
/// every lag in the group covers, then each lag's own scalar tail.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx2")]
unsafe fn lag_group<const K: usize>(w: &[u64], lag0: usize, out: &mut [i64]) {
    use core::arch::x86_64::*;
    let n = w.len();
    let p = w.as_ptr();
    let m = n - (lag0 + K - 1);
    let chunks = m / 4;
    let mut acc = [_mm256_setzero_si256(); K];
    for c in 0..chunks {
        let i = c * 4;
        let y = _mm256_loadu_si256(p.add(i) as *const __m256i);
        for (j, a) in acc.iter_mut().enumerate() {
            let x = _mm256_loadu_si256(p.add(lag0 + j + i) as *const __m256i);
            *a = _mm256_add_epi64(*a, _mm256_mul_epi32(x, y));
        }
    }
    for (j, a) in acc.iter().enumerate() {
        let lag = lag0 + j;
        let mut lanes = [0i64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, *a);
        let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
        for i in chunks * 4..n - lag {
            sum += (w[lag + i] as i32 as i64) * (w[i] as i32 as i64);
        }
        out[lag] = sum;
    }
}

/// The float autocorrelation the integer one replaced — the test oracle.
#[cfg(test)]
fn autocorr_f64(samples: &[i32], win: &[f64], max_order: usize) -> Vec<f64> {
    let x: Vec<f64> = samples
        .iter()
        .zip(win)
        .map(|(&s, &g)| s as f64 * g)
        .collect();
    (0..=max_order)
        .map(|lag| (0..x.len() - lag).map(|i| x[i] * x[i + lag]).sum())
        .collect()
}

/// Levinson-Durbin, error-only pass: fills `errs[i]` with the residual energy
/// after order i+1 and returns how many orders were reachable before the
/// recursion exhausted numerically. No per-order coefficient allocation —
/// [`levinson_coeffs`] re-derives the chosen order's coefficients on demand
/// (O(order²), amortized to nothing next to the O(order·n) autocorrelation).
fn levinson_errs(autoc: &[f64], max_order: usize, errs: &mut [f64; 32]) -> usize {
    let mut lpc = [0.0f64; 32];
    let mut err = autoc[0];
    let mut found = 0usize;
    for i in 0..max_order {
        if err <= 0.0 {
            break; // numerically exhausted; keep the orders found so far
        }
        let mut r = -autoc[i + 1];
        for j in 0..i {
            r -= lpc[j] * autoc[i - j];
        }
        r /= err;
        lpc[i] = r;
        for j in 0..(i / 2) {
            let tmp = lpc[j];
            lpc[j] = tmp + r * lpc[i - 1 - j];
            lpc[i - 1 - j] += r * tmp;
        }
        if i & 1 == 1 {
            lpc[i / 2] += r * lpc[i / 2];
        }
        err *= 1.0 - r * r;
        errs[i] = err;
        found = i + 1;
    }
    found
}

/// Coefficients for one specific order, re-running the recursion. The
/// PREDICTOR coefficients are the negation of the AR solution (libFLAC's
/// `lp_coeff = -lpc`).
fn levinson_coeffs(autoc: &[f64], order: usize) -> Vec<f64> {
    let mut lpc = [0.0f64; 32];
    let mut err = autoc[0];
    for i in 0..order {
        debug_assert!(err > 0.0, "caller checked reachability via levinson_errs");
        let mut r = -autoc[i + 1];
        for j in 0..i {
            r -= lpc[j] * autoc[i - j];
        }
        r /= err;
        lpc[i] = r;
        for j in 0..(i / 2) {
            let tmp = lpc[j];
            lpc[j] = tmp + r * lpc[i - 1 - j];
            lpc[i - 1 - j] += r * tmp;
        }
        if i & 1 == 1 {
            lpc[i / 2] += r * lpc[i / 2];
        }
        err *= 1.0 - r * r;
    }
    lpc[..order].iter().map(|&c| -c).collect()
}

/// Quantize float LPC coefficients to `precision`-bit integers + a NON-negative
/// shift, with libFLAC-style rounding error feedback.
fn quantize_lpc(lpc: &[f64], precision: u32) -> Option<(Vec<i32>, i32)> {
    let cmax = lpc.iter().fold(0.0f64, |m, &c| m.max(c.abs()));
    if !cmax.is_finite() || cmax <= 0.0 {
        return None;
    }
    let exp = math::floor(math::log2(cmax)) as i32 + 1; // frexp exponent of cmax
    let shift = (precision as i32 - exp - 1).clamp(0, 15);
    let qmax = (1i32 << (precision - 1)) - 1;
    let qmin = -(1i32 << (precision - 1));
    let scale = math::exp2(shift as f64);
    let mut error = 0.0f64;
    let mut qlp = Vec::with_capacity(lpc.len());
    for &c in lpc {
        let v = c * scale + error;
        let q = math::round(v).clamp(qmin as f64, qmax as f64);
        error = v - q;
        qlp.push(q as i32);
    }
    if qlp.iter().all(|&q| q == 0) {
        return None; // no predictive power left after quantization
    }
    Some((qlp, shift))
}

/// LPC residual using the quantized coefficients — exact arithmetic the
/// decoder inverts, so it round-trips losslessly.
///
/// The AVX2 path runs the dot products in f64 FMA lanes: with |sample| < 2^25
/// and 15-bit (sign included) coefficients every product is < 2^39 and every
/// partial sum of ≤32 terms is < 2^44 — integers well inside f64's exact range, so
/// FMA ordering cannot change a bit, and `floor(sum · 2^-shift)` equals the
/// arithmetic shift (gated by `lpc_residual_avx2_matches_scalar`).
fn lpc_residual(samples: &[i32], qlp: &[i32], shift: i32, order: usize, peak: u32) -> Vec<i32> {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        // Exactness: the vector path converts the prediction to i32 with
        // `cvtpd_epi32`, which is exact only inside i32 (the scalar form and
        // the decoder wrap). When the block bound `Σ|c|·2^25 >> shift < 2^31`
        // proves every prediction is inside, the plain kernel runs. When it
        // does not -- sharp predictors on high-rate audio fail it while their
        // real predictions stay sample-sized -- the CHECKED kernel tests each
        // vector of four and recomputes the rare one outside i32 the scalar
        // way. Those blocks used to fall to the i64 scalar loop outright (11%
        // of a 96 kHz encode). Two kernels rather than one checked one: the
        // check measured +22 instructions a vector on blocks that need none.
        if samples.len() > order + 8
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            let sum_abs: i64 = qlp[..order].iter().map(|&c| (c as i64).abs()).sum();
            // Integer lanes when every partial sum provably fits i32 (the
            // bound the scalar 32-bit path uses, from the block's own peak):
            // eight samples a vector and one multiply-add per tap, where the
            // f64 path converts and fuses four at a time. 16-bit content
            // almost always qualifies.
            if (sum_abs as u64) << (u32::BITS - peak.leading_zeros()) < 1u64 << 31 {
                // SAFETY: AVX2 checked above; the i32 bound just proven.
                return unsafe { lpc_residual_avx2_i32(samples, qlp, shift, order) };
            }
            // SAFETY: guarded by the runtime AVX2+FMA check.
            return unsafe {
                if (sum_abs << 25) >> shift < (1i64 << 31) {
                    lpc_residual_avx2::<false>(samples, qlp, shift, order)
                } else {
                    lpc_residual_avx2::<true>(samples, qlp, shift, order)
                }
            };
        }
    }
    lpc_residual_scalar(samples, qlp, shift, order, peak)
}

/// The LPC residual in eight i32 lanes, for blocks where
/// `Σ|c| · 2^bits(peak) < 2^31` (checked by the caller): every partial sum is
/// then exact in i32, so `mullo`/`add` equal the i64 arithmetic and the
/// arithmetic shift is the decoder's `sum >> shift`.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx2")]
unsafe fn lpc_residual_avx2_i32(
    samples: &[i32],
    qlp: &[i32],
    shift: i32,
    order: usize,
) -> Vec<i32> {
    use core::arch::x86_64::*;
    let n = samples.len();
    let mut res: Vec<i32> = Vec::with_capacity(n - order);
    let out = res.as_mut_ptr();
    let mut coeffs = [_mm256_setzero_si256(); 32];
    for j in 0..order {
        coeffs[j] = _mm256_set1_epi32(qlp[j]);
    }
    let sh = _mm_cvtsi32_si128(shift);
    let p = samples.as_ptr();
    let mut i = order;
    while i + 8 <= n {
        let mut sum = _mm256_setzero_si256();
        for (j, &c) in coeffs[..order].iter().enumerate() {
            let x = _mm256_loadu_si256(p.add(i - 1 - j) as *const __m256i);
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(x, c));
        }
        let pred = _mm256_sra_epi32(sum, sh);
        let s = _mm256_loadu_si256(p.add(i) as *const __m256i);
        _mm256_storeu_si256(
            out.add(i - order) as *mut __m256i,
            _mm256_sub_epi32(s, pred),
        );
        i += 8;
    }
    for j in i..n {
        let mut sum: i32 = 0;
        for (k, &c) in qlp[..order].iter().enumerate() {
            sum += c * *p.add(j - 1 - k);
        }
        *out.add(j - order) = (*p.add(j)).wrapping_sub(sum >> shift);
    }
    // SAFETY: slots 0..n - order were each written above.
    res.set_len(n - order);
    res
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn lpc_residual_avx2<const CHECK: bool>(
    samples: &[i32],
    qlp: &[i32],
    shift: i32,
    order: usize,
) -> Vec<i32> {
    use core::arch::x86_64::*;
    let n = samples.len();
    let mut res: Vec<i32> = Vec::with_capacity(n - order);
    // Every slot `0..n - order` is written exactly once below, straight into
    // the reserved buffer (one 16-byte store per four), and the length is set
    // at the end. `extend_from_slice` per four paid a capacity check and a
    // copy through a stack array each time. The coefficient broadcasts are
    // made once, not once per tap per vector.
    let out = res.as_mut_ptr();
    let mut coeffs = [_mm256_setzero_pd(); 32];
    for j in 0..order {
        coeffs[j] = _mm256_set1_pd(qlp[j] as f64);
    }
    let scale = _mm256_set1_pd(math::exp2(-(shift as f64))); // 2^-shift, exact
    let hi = _mm256_set1_pd(i32::MAX as f64);
    let lo = _mm256_set1_pd(i32::MIN as f64);
    let p = samples.as_ptr();
    let mut i = order;
    while i + 4 <= n {
        let mut sum = _mm256_setzero_pd();
        for (j, &c) in coeffs[..order].iter().enumerate() {
            let x = _mm256_cvtepi32_pd(_mm_loadu_si128(p.add(i - 1 - j) as *const __m128i));
            sum = _mm256_fmadd_pd(x, c, sum);
        }
        // pred = floor(sum / 2^shift) == sum >> shift (arithmetic).
        let pred = _mm256_floor_pd(_mm256_mul_pd(sum, scale));
        // `cvtpd_epi32` is exact for an integral pred inside i32; outside it
        // the scalar form wraps where the conversion would not, so those four
        // go the scalar way (the decoder's arithmetic).
        if CHECK
            && _mm256_movemask_pd(_mm256_or_pd(
                _mm256_cmp_pd::<_CMP_GT_OQ>(pred, hi),
                _mm256_cmp_pd::<_CMP_LT_OQ>(pred, lo),
            )) != 0
        {
            for j in i..i + 4 {
                let mut sum: i64 = 0;
                for (k, &c) in qlp[..order].iter().enumerate() {
                    sum += c as i64 * *p.add(j - 1 - k) as i64;
                }
                *out.add(j - order) = (*p.add(j)).wrapping_sub((sum >> shift) as i32);
            }
            i += 4;
            continue;
        }
        let predi = _mm256_cvtpd_epi32(pred); // exact: integral and inside i32
        let s = _mm_loadu_si128(p.add(i) as *const __m128i);
        _mm_storeu_si128(out.add(i - order) as *mut __m128i, _mm_sub_epi32(s, predi));
        i += 4;
    }
    for j in i..n {
        let mut sum: i64 = 0;
        for (k, &c) in qlp[..order].iter().enumerate() {
            sum += c as i64 * *p.add(j - 1 - k) as i64;
        }
        *out.add(j - order) = (*p.add(j)).wrapping_sub((sum >> shift) as i32);
    }
    // SAFETY: slots 0..n - order were each written above.
    res.set_len(n - order);
    res
}

/// `peak` is the OR of `|s|` over `samples` (its bit length bounds every
/// sample), as [`autocorrelation`] returns it for the same samples.
fn lpc_residual_scalar(
    samples: &[i32],
    qlp: &[i32],
    shift: i32,
    order: usize,
    peak: u32,
) -> Vec<i32> {
    #[inline(always)]
    fn run<const ORDER: usize>(samples: &[i32], qlp: &[i32], shift: i32) -> Vec<i32> {
        let mut res = Vec::with_capacity(samples.len() - ORDER);
        // i32 coefficients so every product is a 32×32→64 widening multiply
        // (the pattern LLVM lowers to pmuldq lanes).
        let mut coeffs = [0i32; 32];
        coeffs[..ORDER].copy_from_slice(&qlp[..ORDER]);
        for i in ORDER..samples.len() {
            let mut sum: i64 = 0;
            for j in 0..ORDER {
                sum += coeffs[j] as i64 * samples[i - 1 - j] as i64;
            }
            res.push(samples[i].wrapping_sub((sum >> shift) as i32));
        }
        res
    }
    // 32-bit prediction when it provably cannot overflow: every partial sum
    // is bounded by Σ|c|·max|s| < Σ|c|·2^bits(max|s|). When that is below
    // 2^31 the i32 sum equals the i64 one, and so does `sum >> shift` — on a
    // 32-bit core one multiply and one add per tap instead of a widening
    // multiply pair and a 64-bit add with carry. (libFLAC makes the same
    // 32/64-bit split, from bit depths; this one uses the block's own peak,
    // which the autocorrelation already measured — no extra pass.)
    debug_assert_eq!(peak, samples.iter().fold(0, |a, &s| a | s.unsigned_abs()));
    let sum_abs: u64 = qlp[..order].iter().map(|&c| c.unsigned_abs() as u64).sum();
    if sum_abs << (u32::BITS - peak.leading_zeros()) < 1u64 << 31 {
        #[inline(always)]
        fn run32<const ORDER: usize>(samples: &[i32], qlp: &[i32], shift: i32) -> Vec<i32> {
            let mut res = Vec::with_capacity(samples.len() - ORDER);
            let mut coeffs = [0i32; 32];
            coeffs[..ORDER].copy_from_slice(&qlp[..ORDER]);
            for i in ORDER..samples.len() {
                let mut sum: i32 = 0;
                for j in 0..ORDER {
                    sum += coeffs[j] * samples[i - 1 - j];
                }
                res.push(samples[i] - (sum >> shift));
            }
            res
        }
        return match order {
            1 => run32::<1>(samples, qlp, shift),
            2 => run32::<2>(samples, qlp, shift),
            3 => run32::<3>(samples, qlp, shift),
            4 => run32::<4>(samples, qlp, shift),
            5 => run32::<5>(samples, qlp, shift),
            6 => run32::<6>(samples, qlp, shift),
            7 => run32::<7>(samples, qlp, shift),
            8 => run32::<8>(samples, qlp, shift),
            9 => run32::<9>(samples, qlp, shift),
            10 => run32::<10>(samples, qlp, shift),
            11 => run32::<11>(samples, qlp, shift),
            12 => run32::<12>(samples, qlp, shift),
            _ => {
                let mut res = Vec::with_capacity(samples.len() - order);
                for i in order..samples.len() {
                    let mut sum: i32 = 0;
                    for j in 0..order {
                        sum += qlp[j] * samples[i - 1 - j];
                    }
                    res.push(samples[i] - (sum >> shift));
                }
                res
            }
        };
    }
    match order {
        1 => run::<1>(samples, qlp, shift),
        2 => run::<2>(samples, qlp, shift),
        3 => run::<3>(samples, qlp, shift),
        4 => run::<4>(samples, qlp, shift),
        5 => run::<5>(samples, qlp, shift),
        6 => run::<6>(samples, qlp, shift),
        7 => run::<7>(samples, qlp, shift),
        8 => run::<8>(samples, qlp, shift),
        9 => run::<9>(samples, qlp, shift),
        10 => run::<10>(samples, qlp, shift),
        11 => run::<11>(samples, qlp, shift),
        12 => run::<12>(samples, qlp, shift),
        _ => {
            let mut res = Vec::with_capacity(samples.len() - order);
            for i in order..samples.len() {
                let mut sum: i64 = 0;
                for j in 0..order {
                    sum += qlp[j] as i64 * samples[i - 1 - j] as i64;
                }
                res.push(samples[i].wrapping_sub((sum >> shift) as i32));
            }
            res
        }
    }
}

/// A complete LPC subframe candidate + its total bit cost.
struct LpcCandidate {
    order: usize,
    /// Coefficient precision in bits (the subframe's `qlp precision`).
    precision: u32,
    qlp: Vec<i32>,
    shift: i32,
    res: Vec<i32>,
    plan: ResidualPlan,
    bits: u64,
}

/// An estimated (not yet realized) LPC candidate: chosen order + float
/// coefficients + the Levinson bit estimate that ranked it.
#[derive(Clone)]
struct LpcEstimate {
    order: usize,
    coeffs: Vec<f64>,
    est_bits: f64,
    /// OR of `|s|` over the samples this estimate describes (from the
    /// autocorrelation), for the residual's 32-bit bound.
    peak: u32,
}

/// When two windows' estimates are within this relative margin, both are
/// realized exactly and compared — outside it, only the estimated winner is.
/// (The second window wins ~63% of subframes on real music, so it can never
/// be dropped outright; this only prunes the clear-loser realizations.)
const WINDOW_EST_MARGIN: f64 = 0.02;

/// Realize the best LPC subframe from precomputed per-window estimates:
/// realize the estimated winner exactly, and a runner-up only when its
/// estimate is within [`WINDOW_EST_MARGIN`]. None if degenerate.
fn realize_best_window(
    samples: &[i32],
    bps: u32,
    ests: &[Option<LpcEstimate>],
    stats: &mut EncodeStats,
    scratch: &mut EncodeScratch,
) -> Option<LpcCandidate> {
    let best_est = ests
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.as_ref().map(|e| (i, e.est_bits)))
        .min_by(|a, b| a.1.total_cmp(&b.1))?
        .0;

    let mut best: Option<(usize, LpcCandidate)> = None;
    for (widx, est) in ests.iter().enumerate() {
        let Some(est) = est else { continue };
        if widx != best_est {
            let winner = ests[best_est].as_ref().expect("winner exists").est_bits;
            if est.est_bits > winner * (1.0 + WINDOW_EST_MARGIN) {
                continue; // clear loser: skip the expensive realization
            }
        }
        if let Some(c) = realize_lpc(samples, bps, est, stats, &mut *scratch) {
            if best.as_ref().is_none_or(|(_, b)| c.bits < b.bits) {
                best = Some((widx, c));
            }
        }
    }
    let (widx, cand) = best?;
    if widx == 1 {
        stats.lpc_window_second_won += 1;
    }
    Some(cand)
}

/// The cheap half of an LPC candidate: autocorrelation + Levinson + order
/// selection from residual energy. No residual computed yet.
fn lpc_estimate(
    samples: &[i32],
    bps: u32,
    max_order: usize,
    win: Window,
    stats: &mut EncodeStats,
    scratch: &mut EncodeScratch,
) -> Option<LpcEstimate> {
    let n = samples.len();
    let peak = autocorrelation(samples, max_order, win, &mut *scratch);
    let autoc = &scratch.autoc;
    if autoc[0] <= 0.0 {
        return None;
    }
    let mut errs = [0.0f64; 32];
    let found = levinson_errs(autoc, max_order, &mut errs);
    if found == 0 {
        return None;
    }
    if found < max_order {
        stats.lpc_levinson_exhausted += 1;
    }
    // Pick the order from the Levinson residual energy (header cost vs the
    // entropy of a residual with that variance).
    let mut best_idx = 0usize;
    let mut best_est = f64::INFINITY;
    for (idx, &err) in errs[..found].iter().enumerate() {
        let order = idx + 1;
        let var = err / n as f64;
        let bits_per = if var > 0.0 {
            (0.5 * math::log2(var)).max(0.0)
        } else {
            0.0
        };
        let est = order as f64 * (bps + LPC_PRECISION) as f64 + bits_per * (n - order) as f64;
        if est < best_est {
            best_est = est;
            best_idx = idx;
        }
    }
    let coeffs = levinson_coeffs(autoc, best_idx + 1);
    Some(LpcEstimate {
        order: best_idx + 1,
        coeffs,
        est_bits: best_est,
        peak,
    })
}

/// The expensive half: quantize, compute the exact residual, plan partitions.
fn realize_lpc(
    samples: &[i32],
    bps: u32,
    est: &LpcEstimate,
    stats: &mut EncodeStats,
    scratch: &mut EncodeScratch,
) -> Option<LpcCandidate> {
    let n = samples.len();
    let order = est.order;
    let Some((qlp, shift)) = quantize_lpc(&est.coeffs, LPC_PRECISION) else {
        stats.lpc_quantize_failed += 1;
        return None;
    };
    let res = lpc_residual(samples, &qlp, shift, order, est.peak);
    let plan = plan_partitions(&res, n, order, scratch);
    // hdr(8) + warm-up + precision(4) + shift(5) + coeffs + residual hdr(6) + body.
    let bits =
        8 + order as u64 * bps as u64 + 4 + 5 + order as u64 * LPC_PRECISION as u64 + 6 + plan.bits;
    Some(LpcCandidate {
        order,
        precision: LPC_PRECISION,
        qlp,
        shift,
        res,
        plan,
        bits,
    })
}

// ---------------------------------------------------------------------------
// Subframe selection
// ---------------------------------------------------------------------------

/// One-pass FIXED-order estimator: |residual| sums for orders 0..=4 via the
/// direct difference formulas — no allocation, single sweep (the libFLAC
/// order-selection method). Returns the chosen order and its |residual| sum.
/// Integer math, so the AVX2 kernel is exact (gated by
/// `fixed_sums_avx2_matches_scalar`).
fn fixed_order_estimate(samples: &[i32], bps: u32) -> (usize, u64) {
    // The AVX2 lane schedule needs |order-4 residual| < 2^(bps + 3) in i32
    // lanes; wider input takes `fixed_sums_wide` (i64) and never gets here.
    debug_assert!(bps <= 25, "wide input uses fixed_sums_wide");
    let n = samples.len();
    let max_order = 4.min(n.saturating_sub(1));
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    let sums = if n >= 16 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by the runtime AVX2 check.
        unsafe { fixed_sums_avx2(samples, bps) }
    } else {
        fixed_sums_scalar(samples)
    };
    #[cfg(not(all(target_arch = "x86_64", feature = "std")))]
    let sums = {
        let _ = bps; // only the AVX2 kernel's lane-width schedule reads it
        fixed_sums_scalar(samples)
    };

    let mut best = 0usize;
    for order in 1..=max_order {
        if sums[order] < sums[best] {
            best = order;
        }
    }
    (best, sums[best])
}

fn fixed_sums_scalar(samples: &[i32]) -> [u64; 5] {
    let n = samples.len();
    let mut sums = [0u64; 5];
    // Ramp-in: orders become defined at i >= order.
    for i in 0..n.min(4) {
        let s = |j: usize| samples[i - j] as i64;
        sums[0] += s(0).unsigned_abs();
        if i >= 1 {
            sums[1] += (s(0) - s(1)).unsigned_abs();
        }
        if i >= 2 {
            sums[2] += (s(0) - 2 * s(1) + s(2)).unsigned_abs();
        }
        if i >= 3 {
            sums[3] += (s(0) - 3 * s(1) + 3 * s(2) - s(3)).unsigned_abs();
        }
    }
    if n <= 4 {
        return sums;
    }
    // Steady state: one pass with the previous sample and the previous
    // order-1..3 differences in registers; the order-k difference is the first
    // difference of the order-(k-1) one, which is the direct formula exactly,
    // as integers. |sample| < 2^25 (24-bit + side), so the order-k difference
    // is below 2^(25+k) <= 2^29: exact in i32, and four samples' worth fits a
    // u32 partial sum, folded into the u64 totals once per 4 samples. (On a
    // 32-bit core: one-instruction subtracts and `abs` instead of 64-bit
    // multiply/add/abs sequences, and no per-sample call.)
    let mut prev = samples[3];
    let mut d1 = samples[3] - samples[2];
    let mut d2 = d1 - (samples[2] - samples[1]);
    let mut d3 = d2 - ((samples[2] - samples[1]) - (samples[1] - samples[0]));
    macro_rules! step {
        ($s0:expr, $p:ident) => {{
            let s0: i32 = $s0;
            let e1 = s0 - prev;
            let e2 = e1 - d1;
            let e3 = e2 - d2;
            let e4 = e3 - d3;
            $p[0] += s0.unsigned_abs();
            $p[1] += e1.unsigned_abs();
            $p[2] += e2.unsigned_abs();
            $p[3] += e3.unsigned_abs();
            $p[4] += e4.unsigned_abs();
            prev = s0;
            d1 = e1;
            d2 = e2;
            d3 = e3;
        }};
    }
    let mut chunks = samples[4..].chunks_exact(4);
    for c in &mut chunks {
        let mut p = [0u32; 5];
        step!(c[0], p);
        step!(c[1], p);
        step!(c[2], p);
        step!(c[3], p);
        for (s, v) in sums.iter_mut().zip(p) {
            *s += v as u64;
        }
    }
    let mut p = [0u32; 5];
    for &s0 in chunks.remainder() {
        step!(s0, p);
    }
    for (s, v) in sums.iter_mut().zip(p) {
        *s += v as u64;
    }
    sums
}

/// 8-lane i32 differences (nested first-differences give every order), then
/// abs + widening u64 accumulation. Bounded: |sample| < 2^25 (24-bit + side),
/// order-4 coefficient sum 16 ⇒ every intermediate fits i32.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[target_feature(enable = "avx2")]
unsafe fn fixed_sums_avx2(samples: &[i32], bps: u32) -> [u64; 5] {
    use core::arch::x86_64::*;
    let n = samples.len();
    debug_assert!(n >= 16);

    // |residual| of order k is below 2^k * 2^(bps - 1), so order 4's is below
    // 2^(bps + 3) and `m` of them sum inside a u32 lane when
    // m <= 2^(29 - bps). The absolute values accumulate in u32 lanes -- two
    // instructions per order per vector -- and widen to the u64 totals once
    // per `m` vectors: every 8192 at 16-bit (never, within a block), every 32
    // at 24. Widening every vector, as before, was six per order.
    let m = 1usize << 29u32.saturating_sub(bps).min(20);
    #[inline(always)]
    unsafe fn widen(acc: &mut __m256i, a: __m256i) {
        let lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(a));
        let hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(a, 1));
        *acc = _mm256_add_epi64(*acc, _mm256_add_epi64(lo, hi));
    }
    #[inline(always)]
    unsafe fn hsum(acc: __m256i) -> u64 {
        let lo = _mm256_castsi256_si128(acc);
        let hi = _mm256_extracti128_si256(acc, 1);
        let s = _mm_add_epi64(lo, hi);
        (_mm_cvtsi128_si64(s) as u64).wrapping_add(_mm_extract_epi64(s, 1) as u64)
    }

    let p = samples.as_ptr();
    let mut a0 = _mm256_setzero_si256();
    let mut a1 = _mm256_setzero_si256();
    let mut a2 = _mm256_setzero_si256();
    let mut a3 = _mm256_setzero_si256();
    let mut a4 = _mm256_setzero_si256();
    let mut i = 4usize;
    while i + 8 <= n {
        let mut b = [_mm256_setzero_si256(); 5];
        // A counted trip, not a two-condition test per vector: the bound
        // check (`cnt < m && i + 8 <= n`) was a quarter of the loop body.
        for _ in 0..m.min((n - i) / 8) {
            let s0 = _mm256_loadu_si256(p.add(i) as *const __m256i);
            let s1 = _mm256_loadu_si256(p.add(i - 1) as *const __m256i);
            let s2 = _mm256_loadu_si256(p.add(i - 2) as *const __m256i);
            let s3 = _mm256_loadu_si256(p.add(i - 3) as *const __m256i);
            let s4 = _mm256_loadu_si256(p.add(i - 4) as *const __m256i);
            let r1 = _mm256_sub_epi32(s0, s1);
            let d1 = _mm256_sub_epi32(s1, s2); // r1 shifted one sample back
            let r2 = _mm256_sub_epi32(r1, d1);
            let d2 = _mm256_sub_epi32(d1, _mm256_sub_epi32(s2, s3)); // r2 shifted
            let r3 = _mm256_sub_epi32(r2, d2);
            let e2 = _mm256_sub_epi32(_mm256_sub_epi32(s2, s3), _mm256_sub_epi32(s3, s4));
            let d3 = _mm256_sub_epi32(d2, e2); // r3 shifted
            let r4 = _mm256_sub_epi32(r3, d3);
            b[0] = _mm256_add_epi32(b[0], _mm256_abs_epi32(s0));
            b[1] = _mm256_add_epi32(b[1], _mm256_abs_epi32(r1));
            b[2] = _mm256_add_epi32(b[2], _mm256_abs_epi32(r2));
            b[3] = _mm256_add_epi32(b[3], _mm256_abs_epi32(r3));
            b[4] = _mm256_add_epi32(b[4], _mm256_abs_epi32(r4));
            i += 8;
        }
        widen(&mut a0, b[0]);
        widen(&mut a1, b[1]);
        widen(&mut a2, b[2]);
        widen(&mut a3, b[3]);
        widen(&mut a4, b[4]);
    }
    let mut sums = [hsum(a0), hsum(a1), hsum(a2), hsum(a3), hsum(a4)];

    // Head (order-0 covers 0..4 + ramp-in of orders 1..3) and tail, scalar.
    for &v in &samples[..4.min(n)] {
        sums[0] += (v as i64).unsigned_abs();
    }
    for j in 1..n.min(4) {
        let s = |k: usize| samples[j - k] as i64;
        sums[1] += (s(0) - s(1)).unsigned_abs();
        if j >= 2 {
            sums[2] += (s(0) - 2 * s(1) + s(2)).unsigned_abs();
        }
        if j >= 3 {
            sums[3] += (s(0) - 3 * s(1) + 3 * s(2) - s(3)).unsigned_abs();
        }
    }
    for j in i..n {
        let s0 = samples[j] as i64;
        let s1 = samples[j - 1] as i64;
        let s2 = samples[j - 2] as i64;
        let s3 = samples[j - 3] as i64;
        let s4 = samples[j - 4] as i64;
        sums[0] += s0.unsigned_abs();
        sums[1] += (s0 - s1).unsigned_abs();
        sums[2] += (s0 - 2 * s1 + s2).unsigned_abs();
        sums[3] += (s0 - 3 * s1 + 3 * s2 - s3).unsigned_abs();
        sums[4] += (s0 - 4 * s1 + 6 * s2 - 4 * s3 + s4).unsigned_abs();
    }
    sums
}

/// Estimated single-partition Rice bit cost from a |residual| sum: pick the
/// parameter from the folded mean and price `Σ(u>>k) ≈ (Σu)>>k` (error < cnt).
fn rice_bits_estimate(abs_sum: u64, cnt: u64) -> u64 {
    if cnt == 0 {
        return 0;
    }
    let usum = abs_sum.saturating_mul(2); // zigzag(v) ∈ {2|v|, 2|v|−1}
    let mean = usum / cnt;
    let k = if mean > 0 {
        63 - mean.leading_zeros()
    } else {
        0
    }
    .min(RICE_KMAX as u32);
    // Check k−1, k, k+1 — the mean-derived parameter is within one of optimal.
    let mut best = u64::MAX;
    for kk in k.saturating_sub(1)..=(k + 1).min(RICE_KMAX as u32) {
        let bits = cnt * (1 + kk as u64) + (usum >> kk);
        best = best.min(bits);
    }
    best
}

/// The FIXED residual of one order via its direct formula — one vectorizable
/// pass, i32 arithmetic (bounded: |sample| < 2^25, coefficient sum ≤ 16 ⇒
/// |residual| < 2^30).
fn fixed_residual(samples: &[i32], order: usize) -> Vec<i32> {
    let n = samples.len();
    let mut res = Vec::with_capacity(n - order);
    match order {
        0 => res.extend_from_slice(samples),
        1 => {
            for i in 1..n {
                res.push(samples[i].wrapping_sub(samples[i - 1]));
            }
        }
        2 => {
            for i in 2..n {
                res.push(samples[i] - 2 * samples[i - 1] + samples[i - 2]);
            }
        }
        3 => {
            for i in 3..n {
                res.push(samples[i] - 3 * samples[i - 1] + 3 * samples[i - 2] - samples[i - 3]);
            }
        }
        4 => {
            for i in 4..n {
                res.push(
                    samples[i] - 4 * samples[i - 1] + 6 * samples[i - 2] - 4 * samples[i - 3]
                        + samples[i - 4],
                );
            }
        }
        _ => unreachable!("fixed order 0..=4"),
    }
    res
}

/// One coded subframe: its (possibly wasted-shifted) samples, effective bit
/// depth, and chosen encoding.
type Subframe<'a> = (Cow<'a, [i32]>, u32, SubframeChoice);

/// The chosen subframe encoding for a channel + its bit cost.
struct SubframeChoice {
    bits: u64,
    /// Trailing zero bits shifted out of every sample of this subframe
    /// (FLAC's wasted-bits field). The analysis ran on the SHIFTED samples at
    /// `bps - wasted`; `bits` includes the unary wasted-count header cost.
    wasted: u32,
    kind: SubframeKind,
}

/// Trailing zero bits common to every sample of the block (0 for all-zero
/// input — that's the CONSTANT path). This is what ffmpeg/libFLAC strip on
/// 16-bit-content-in-24-bit-container material, worth 8 bits/sample there.
fn detect_wasted(samples: &[i32], bps: u32) -> u32 {
    let mut acc = 0i32;
    for &v in samples {
        acc |= v;
        if acc & 1 != 0 {
            return 0; // early out: any odd sample kills the shift
        }
    }
    if acc == 0 {
        return 0;
    }
    (acc.trailing_zeros()).min(bps - 1)
}

enum SubframeKind {
    Constant(i32),
    Verbatim,
    Fixed {
        order: usize,
        res: Vec<i32>,
        plan: ResidualPlan,
    },
    Lpc(Box<LpcCandidate>),
}

/// The cheap phase of one arm's analysis: constant detection, LPC estimates
/// for every window, the fixed-order estimate — everything short of residual
/// realization. `est_bits` is the arm's estimated subframe cost, used for
/// stereo-mode gating before any expensive realization happens.
#[derive(Default)]
struct ArmEstimate {
    constant: Option<i32>,
    ests: Vec<Option<LpcEstimate>>,
    est_bits: u64,
}

/// One arm's analysis input: samples with any wasted bits already shifted
/// out, the effective bit depth, and the wasted count for the header.
struct ArmInput<'a> {
    samples: alloc::borrow::Cow<'a, [i32]>,
    /// Effective coded depth: nominal bps − wasted.
    ebps: u32,
    wasted: u32,
}

impl<'a> ArmInput<'a> {
    /// Detect trailing-zero (wasted) bits and shift them out.
    fn prepare(samples: &'a [i32], bps: u32) -> ArmInput<'a> {
        let wasted = detect_wasted(samples, bps);
        if wasted == 0 {
            ArmInput {
                samples: alloc::borrow::Cow::Borrowed(samples),
                ebps: bps,
                wasted: 0,
            }
        } else {
            ArmInput {
                samples: alloc::borrow::Cow::Owned(samples.iter().map(|&v| v >> wasted).collect()),
                ebps: bps - wasted,
                wasted,
            }
        }
    }

    /// [`ArmInput::prepare`] for computed samples the arm can own outright
    /// (mid, side): with no wasted bits the vector moves in, no copy.
    fn prepare_owned(samples: Vec<i32>, bps: u32) -> ArmInput<'a> {
        let wasted = detect_wasted(&samples, bps);
        if wasted == 0 {
            return ArmInput {
                samples: alloc::borrow::Cow::Owned(samples),
                ebps: bps,
                wasted: 0,
            };
        }
        ArmInput {
            samples: alloc::borrow::Cow::Owned(samples.iter().map(|&v| v >> wasted).collect()),
            ebps: bps - wasted,
            wasted,
        }
    }

    /// Hand back the samples without materializing them: a borrow stays a
    /// borrow (no copy) and an owned shift moves out. The caller must keep the
    /// borrowed source alive until the subframe is written.
    fn into_cow(self) -> Cow<'a, [i32]> {
        self.samples
    }
}

fn estimate_arm(
    arm: &ArmInput<'_>,
    max_lpc_order: usize,
    wins: &WindowCache,
    stats: &mut EncodeStats,
    scratch: &mut EncodeScratch,
) -> ArmEstimate {
    let samples: &[i32] = &arm.samples;
    let bps = arm.ebps;
    let n = samples.len();
    if samples.iter().all(|&s| s == samples[0]) {
        return ArmEstimate {
            constant: Some(samples[0]),
            ests: Vec::new(),
            est_bits: 8 + arm.wasted as u64 + bps as u64,
        };
    }
    let max_order = max_lpc_order.min(n / 2);
    // Phase 1 estimates only the FIRST window — arm/mode ranking correlates
    // strongly across windows, so the second window's estimate is deferred to
    // realization (realize_arm), skipping two autocorrelations per pruned arm.
    let ests: Vec<Option<LpcEstimate>> = if max_order >= 1 {
        debug_assert_eq!(wins.n, n, "window cache not sized for this block");
        // Sized for every window: realize_arm appends the remaining windows'
        // estimates to this same Vec, so reserving the full window count here
        // saves it a re-grow.
        let mut v = Vec::with_capacity(WINDOW_ALPHAS.len());
        v.push(lpc_estimate(
            samples,
            bps,
            max_order,
            wins.get(0),
            stats,
            scratch,
        ));
        v
    } else {
        Vec::new()
    };
    let lpc_est = ests
        .iter()
        .flatten()
        .map(|e| e.est_bits)
        .fold(f64::INFINITY, f64::min);
    let verbatim = 8 + n as u64 * bps as u64;
    // The FIXED estimate is computed lazily in realize_arm — for arm RANKING
    // the LPC estimate suffices (FIXED wins only degenerate content, and pure
    // silence is already caught by the constant check above).
    let est_bits = if lpc_est.is_finite() {
        (lpc_est as u64).min(verbatim)
    } else {
        let (fx_order, fx_abs) = fixed_order_estimate(samples, bps);
        let fx_est = 8
            + fx_order as u64 * bps as u64
            + 6
            + rice_bits_estimate(fx_abs, (n - fx_order) as u64);
        fx_est.min(verbatim)
    };
    ArmEstimate {
        constant: None,
        ests,
        est_bits: est_bits + arm.wasted as u64,
    }
}

/// The expensive phase: realize the estimated LPC winner (and close runner-up
/// windows), the estimate-gated FIXED plan, and pick the cheapest subframe.
/// The remaining windows' estimates (deferred by phase 1) are computed here.
fn realize_arm(
    arm: &ArmInput<'_>,
    est: ArmEstimate,
    max_lpc_order: usize,
    wins: &WindowCache,
    stats: &mut EncodeStats,
    scratch: &mut EncodeScratch,
) -> SubframeChoice {
    let samples: &[i32] = &arm.samples;
    let bps = arm.ebps;
    // The wasted-bits header cost (unary count) rides on every kind's bits so
    // stereo-mode comparisons stay honest.
    let wb = arm.wasted as u64;
    let n = samples.len();
    if let Some(v) = est.constant {
        return SubframeChoice {
            bits: 8 + wb + bps as u64,
            wasted: arm.wasted,
            kind: SubframeKind::Constant(v),
        };
    }

    // Complete the window-estimate set (phase 1 only did window 0). Take the
    // phase-1 estimates by value (they are no longer needed by the caller) so
    // the window-0 estimate and its coefficients are not re-cloned here.
    let max_order = max_lpc_order.min(n / 2);
    let mut all_ests: Vec<Option<LpcEstimate>> = est.ests;
    if max_order >= 1 {
        for k in all_ests.len()..wins.count() {
            all_ests.push(lpc_estimate(
                samples,
                bps,
                max_order,
                wins.get(k),
                stats,
                &mut *scratch,
            ));
        }
    }
    let lpc = realize_best_window(samples, bps, &all_ests, stats, &mut *scratch);
    let lpc_bits = lpc.as_ref().map_or(u64::MAX, |c| c.bits.saturating_add(wb));

    // FIXED: one-pass order estimate, then the exact residual + partition
    // plan only when the estimate says FIXED could still beat the realized
    // LPC candidate (a wide 10% margin — partitioning can undercut the
    // single-partition estimate). LPC wins ~99.6% of real subframes, so this
    // skips the second-most-expensive per-arm stage almost always.
    let (fx_order, fx_abs) = fixed_order_estimate(samples, bps);
    let fx_est =
        8 + fx_order as u64 * bps as u64 + 6 + rice_bits_estimate(fx_abs, (n - fx_order) as u64);
    let fixed = if lpc.is_none() || fx_est <= lpc_bits.saturating_add(lpc_bits / 10) {
        let fx_res = fixed_residual(samples, fx_order);
        let fx_plan = plan_partitions(&fx_res, n, fx_order, &mut *scratch);
        let fixed_bits = 8 + wb + fx_order as u64 * bps as u64 + 6 + fx_plan.bits;
        Some((fx_res, fx_plan, fixed_bits))
    } else {
        None
    };
    let fixed_bits = fixed.as_ref().map_or(u64::MAX, |f| f.2);

    let verbatim_bits = 8 + wb + n as u64 * bps as u64;

    if lpc_bits <= fixed_bits && lpc_bits <= verbatim_bits {
        SubframeChoice {
            bits: lpc_bits,
            wasted: arm.wasted,
            kind: SubframeKind::Lpc(Box::new(lpc.unwrap())),
        }
    } else if let Some((fx_res, fx_plan, fixed_bits)) = fixed {
        if fixed_bits <= verbatim_bits {
            SubframeChoice {
                bits: fixed_bits,
                wasted: arm.wasted,
                kind: SubframeKind::Fixed {
                    order: fx_order,
                    res: fx_res,
                    plan: fx_plan,
                },
            }
        } else {
            SubframeChoice {
                bits: verbatim_bits,
                wasted: arm.wasted,
                kind: SubframeKind::Verbatim,
            }
        }
    } else {
        SubframeChoice {
            bits: verbatim_bits,
            wasted: arm.wasted,
            kind: SubframeKind::Verbatim,
        }
    }
}

/// Write the shared subframe-header prefix: padding bit, 6-bit type, and the
/// wasted-bits flag (+ unary count).
fn write_subframe_header(bw: &mut BitWriter, type_code: u64, wasted: u32, stats: &mut EncodeStats) {
    bw.write_bits(0, 1);
    bw.write_bits(type_code, 6);
    if wasted == 0 {
        bw.write_bits(0, 1);
    } else {
        stats.sub_wasted_bits += 1;
        bw.write_bits(1, 1);
        bw.write_zeros(wasted - 1); // unary: (wasted-1) zeros then a 1
        bw.write_bits(1, 1);
    }
}

fn write_subframe_from(
    bw: &mut BitWriter,
    samples: &[i32],
    bps: u32,
    choice: &SubframeChoice,
    stats: &mut EncodeStats,
) {
    match &choice.kind {
        SubframeKind::Constant(v) => {
            stats.sub_constant += 1;
            write_subframe_header(bw, 0b000000, choice.wasted, stats);
            bw.write_signed(*v as i64, bps);
        }
        SubframeKind::Verbatim => {
            stats.sub_verbatim += 1;
            write_subframe_header(bw, 0b000001, choice.wasted, stats);
            for &s in samples {
                bw.write_signed(s as i64, bps);
            }
        }
        SubframeKind::Fixed { order, res, plan } => {
            stats.sub_fixed += 1;
            stats.fixed_orders[*order] += 1;
            stats.partition_orders[plan.partition_order as usize] += 1;
            // FIXED, order in low 3 bits
            write_subframe_header(bw, 0b001000 | *order as u64, choice.wasted, stats);
            for &s in &samples[..*order] {
                bw.write_signed(s as i64, bps);
            }
            bw.write_bits(plan.method as u64, 2);
            bw.write_bits(plan.partition_order as u64, 4);
            write_partitioned_residual(bw, res, samples.len(), *order, plan);
        }
        SubframeKind::Lpc(c) => {
            stats.sub_lpc += 1;
            stats.partition_orders[c.plan.partition_order as usize] += 1;
            // LPC, (order-1) in low 5 bits
            write_subframe_header(bw, 0b100000 | (c.order as u64 - 1), choice.wasted, stats);
            for &s in &samples[..c.order] {
                bw.write_signed(s as i64, bps); // warm-up
            }
            bw.write_bits((c.precision - 1) as u64, 4); // qlp precision - 1
            bw.write_bits(c.shift as u64 & 0x1F, 5); // shift (non-negative, 5-bit)
            for &q in &c.qlp {
                bw.write_signed(q as i64, c.precision); // coefficients, qlp[0] first
            }
            bw.write_bits(c.plan.method as u64, 2);
            bw.write_bits(c.plan.partition_order as u64, 4);
            write_partitioned_residual(bw, &c.res, samples.len(), c.order, &c.plan);
        }
    }
}

// ---------------------------------------------------------------------------
// Wide (32-bit) subframes
// ---------------------------------------------------------------------------
//
// Above 24 bits the narrow kernels' bounds (|sample| < 2^25, i32 residual
// arithmetic, the AVX2 exactness guards) no longer hold, so the wide path has
// its own residual code: predictions in i64, and a candidate is eligible only
// when every residual fits 30 bits (|r| < 2^30), which keeps the i32 Rice
// machinery (zigzag, shifted sums, `write_rice`) exact. The analysis that is
// bit-depth independent is shared: windowed autocorrelation (exact for
// |sample| <= 2^31, see `autocorrelation`), Levinson, quantization and the
// partition planner. Nothing here runs for bps <= 24.

/// Highest LPC order the wide path searches. 32-bit streams are never
/// subset, so FLAC's 32 would be legal, but on a 384 kHz / 32-bit track 16
/// saved only 0.5% for 25% more encode time.
const WIDE_MAX_ORDER: usize = LPC_MAX_ORDER;

/// Largest residual magnitude (exclusive) the wide path codes.
const WIDE_RES_LIMIT: u64 = 1 << 30;

/// |residual| sums of the FIXED orders 0..=4, in i64 (the narrow
/// [`fixed_sums_scalar`] runs its steady state in i32).
fn fixed_sums_wide(samples: &[i32]) -> [u64; 5] {
    let mut sums = [0u64; 5];
    let x = |i: usize| samples[i] as i64;
    for i in 0..samples.len() {
        sums[0] += x(i).unsigned_abs();
        if i >= 1 {
            sums[1] += (x(i) - x(i - 1)).unsigned_abs();
        }
        if i >= 2 {
            sums[2] += (x(i) - 2 * x(i - 1) + x(i - 2)).unsigned_abs();
        }
        if i >= 3 {
            sums[3] += (x(i) - 3 * x(i - 1) + 3 * x(i - 2) - x(i - 3)).unsigned_abs();
        }
        if i >= 4 {
            sums[4] +=
                (x(i) - 4 * x(i - 1) + 6 * x(i - 2) - 4 * x(i - 3) + x(i - 4)).unsigned_abs();
        }
    }
    sums
}

/// The FIXED residual of one order in i64, or `None` if any residual reaches
/// [`WIDE_RES_LIMIT`].
fn fixed_residual_wide(samples: &[i32], order: usize) -> Option<Vec<i32>> {
    let x = |i: usize| samples[i] as i64;
    let mut res = Vec::with_capacity(samples.len() - order);
    for i in order..samples.len() {
        let r = match order {
            0 => x(i),
            1 => x(i) - x(i - 1),
            2 => x(i) - 2 * x(i - 1) + x(i - 2),
            3 => x(i) - 3 * x(i - 1) + 3 * x(i - 2) - x(i - 3),
            _ => x(i) - 4 * x(i - 1) + 6 * x(i - 2) - 4 * x(i - 3) + x(i - 4),
        };
        if r.unsigned_abs() >= WIDE_RES_LIMIT {
            return None;
        }
        res.push(r as i32);
    }
    Some(res)
}

/// The LPC residual in i64, or `None` if any residual reaches
/// [`WIDE_RES_LIMIT`]. Every product is below 2^31 · 2^13 and a sum of at
/// most 32 of them below 2^49, so the i64 accumulation is exact; the decoder
/// forms the same `sum >> shift`.
fn lpc_residual_wide(samples: &[i32], qlp: &[i32], shift: i32) -> Option<Vec<i32>> {
    #[inline(always)]
    fn run<const ORDER: usize>(samples: &[i32], qlp: &[i32], shift: i32) -> Option<Vec<i32>> {
        let mut coeffs = [0i64; 32];
        for (c, &q) in coeffs.iter_mut().zip(qlp) {
            *c = q as i64;
        }
        let mut res = Vec::with_capacity(samples.len() - ORDER);
        for i in ORDER..samples.len() {
            let mut sum = 0i64;
            for j in 0..ORDER {
                sum += coeffs[j] * samples[i - 1 - j] as i64;
            }
            let r = samples[i] as i64 - (sum >> shift);
            if r.unsigned_abs() >= WIDE_RES_LIMIT {
                return None;
            }
            res.push(r as i32);
        }
        Some(res)
    }
    match qlp.len() {
        1 => run::<1>(samples, qlp, shift),
        2 => run::<2>(samples, qlp, shift),
        3 => run::<3>(samples, qlp, shift),
        4 => run::<4>(samples, qlp, shift),
        5 => run::<5>(samples, qlp, shift),
        6 => run::<6>(samples, qlp, shift),
        7 => run::<7>(samples, qlp, shift),
        8 => run::<8>(samples, qlp, shift),
        9 => run::<9>(samples, qlp, shift),
        10 => run::<10>(samples, qlp, shift),
        11 => run::<11>(samples, qlp, shift),
        12 => run::<12>(samples, qlp, shift),
        order => {
            let mut res = Vec::with_capacity(samples.len() - order);
            for i in order..samples.len() {
                let sum: i64 = qlp
                    .iter()
                    .enumerate()
                    .map(|(j, &c)| c as i64 * samples[i - 1 - j] as i64)
                    .sum();
                let r = samples[i] as i64 - (sum >> shift);
                if r.unsigned_abs() >= WIDE_RES_LIMIT {
                    return None;
                }
                res.push(r as i32);
            }
            Some(res)
        }
    }
}

/// A quantized wide LPC predictor with its exact residual, not yet
/// partition-planned; `est` is its Rice-estimated subframe size.
struct WideResidual {
    order: usize,
    qlp: Vec<i32>,
    shift: i32,
    res: Vec<i32>,
    est: u64,
}

fn wide_lpc_header_bits(order: usize, bps: u32) -> u64 {
    // hdr(8) + warm-up + precision(4) + shift(5) + coeffs + residual hdr(6).
    8 + order as u64 * bps as u64 + 4 + 5 + order as u64 * LPC_PRECISION as u64 + 6
}

impl WideResidual {
    fn plan(self, n: usize, bps: u32, scratch: &mut EncodeScratch) -> LpcCandidate {
        let plan = plan_partitions(&self.res, n, self.order, scratch);
        LpcCandidate {
            order: self.order,
            precision: LPC_PRECISION,
            bits: wide_lpc_header_bits(self.order, bps) + plan.bits,
            qlp: self.qlp,
            shift: self.shift,
            res: self.res,
            plan,
        }
    }
}

/// How many of [`SEG_WINDOWS`] the wide path tries at a compression level
/// (given as its LPC order cap).
fn wide_seg_windows(max_lpc_order: usize) -> usize {
    if max_lpc_order >= 12 {
        SEG_WINDOWS.len() // partial (2), (3) and punchout (3)
    } else if max_lpc_order >= 8 {
        5 // partial (2) and (3)
    } else {
        0
    }
}

/// Windowed autocorrelations (lags `0..=max_order`) of every window the wide
/// path tries: the two full-block Tukey windows, then the first `seg_windows`
/// of [`SEG_WINDOWS`] (when the block is large enough). Each piece's
/// autocorrelation is computed once and shared by the windows containing it.
fn wide_autocs(
    samples: &[i32],
    max_order: usize,
    seg_windows: usize,
    wins: &WindowCache,
    scratch: &mut EncodeScratch,
) -> Vec<[f64; WIDE_MAX_ORDER + 1]> {
    let n = samples.len();
    let set = wins.wide();
    let lags = max_order + 1;
    let mut out = Vec::with_capacity(2 + SEG_WINDOWS.len());
    for k in 0..WINDOW_ALPHAS.len() {
        autocorrelation(samples, max_order, set.full[k].view(), &mut *scratch);
        let mut a = [0.0f64; WIDE_MAX_ORDER + 1];
        a[..lags].copy_from_slice(&scratch.autoc[..lags]);
        out.push(a);
    }
    if set.pieces.is_empty() {
        return out;
    }
    let mut piece_autoc: [Option<[f64; WIDE_MAX_ORDER + 1]>; PIECES.len()] = [None; PIECES.len()];
    for parts in &SEG_WINDOWS[..seg_windows.min(SEG_WINDOWS.len())] {
        let mut autoc = [0.0f64; WIDE_MAX_ORDER + 1];
        for &p in *parts {
            let pa = piece_autoc[p].get_or_insert_with(|| {
                let (d, i, j) = PIECES[p];
                let seg = &samples[n * i / d..n * j / d];
                autocorrelation(seg, max_order, set.pieces[p].view(), &mut *scratch);
                let mut a = [0.0f64; WIDE_MAX_ORDER + 1];
                a[..lags].copy_from_slice(&scratch.autoc[..lags]);
                a
            });
            for (acc, &v) in autoc[..lags].iter_mut().zip(&pa[..lags]) {
                *acc += v;
            }
        }
        out.push(autoc);
    }
    out
}

/// Levinson-Durbin keeping every order's predictor coefficients:
/// `coeffs[o - 1][..o]` for orders `1..=found` (returned). Same recursion as
/// [`levinson_errs`] / [`levinson_coeffs`].
fn levinson_all(
    autoc: &[f64],
    max_order: usize,
    coeffs: &mut [[f64; WIDE_MAX_ORDER]; WIDE_MAX_ORDER],
) -> usize {
    let mut lpc = [0.0f64; WIDE_MAX_ORDER];
    let mut err = autoc[0];
    let mut found = 0usize;
    for i in 0..max_order {
        if err <= 0.0 {
            break;
        }
        let mut r = -autoc[i + 1];
        for j in 0..i {
            r -= lpc[j] * autoc[i - j];
        }
        r /= err;
        lpc[i] = r;
        for j in 0..(i / 2) {
            let tmp = lpc[j];
            lpc[j] = tmp + r * lpc[i - 1 - j];
            lpc[i - 1 - j] += r * tmp;
        }
        if i & 1 == 1 {
            lpc[i / 2] += r * lpc[i / 2];
        }
        err *= 1.0 - r * r;
        for (c, &l) in coeffs[i].iter_mut().zip(&lpc[..=i]) {
            *c = -l;
        }
        found = i + 1;
    }
    found
}

/// Exact sums for the residual energy of any predictor of order `p <= maxp`
/// on one block: `Σ_{t=p}^{n-1} (Σ_i a_i x_{t-i})² = Σ_ij a_i a_j C_ij` with
/// `C_ij = Σ_{t=p}^{n-1} x_{t-i} x_{t-j}`, which is the full lag sum `S_k`
/// (`k = |i - j|`) minus the products before `t = p` (`head`) and past the
/// block's end (`tail`). All i128 and exact: on oversampled content the
/// residual energy is ~2^-40 of the signal's, a cancellation an f64 would not
/// survive.
struct WideCov {
    /// `maxp + 1`, the row length of `head` and `tail`.
    w: usize,
    /// `S_k = Σ_{u=k}^{n-1} x_u x_{u-k}`.
    lag: Vec<i128>,
    /// `head[k·w + m] = Σ_{u=k}^{m-1} x_u x_{u-k}` (`k <= m <= maxp`).
    head: Vec<i128>,
    /// `tail[k·w + i] = Σ_{u=n-i}^{n-1} x_u x_{u-k}` (`i <= maxp`).
    tail: Vec<i128>,
}

impl WideCov {
    /// Needs `x.len() >= 2 * maxp`.
    fn new(x: &[i32], maxp: usize) -> WideCov {
        let n = x.len();
        debug_assert!(n >= 2 * maxp);
        let w = maxp + 1;
        // |x| <= 2^31: each product is at most 2^62 and exact in i64; sums
        // of them are taken in i128.
        let prod = |u: usize, k: usize| (x[u] as i64 * x[u - k] as i64) as i128;
        let mut c = WideCov {
            w,
            lag: vec![0; w],
            head: vec![0; w * w],
            tail: vec![0; w * w],
        };
        for k in 0..=maxp {
            c.lag[k] = x[k..]
                .iter()
                .zip(&x[..n - k])
                .map(|(&a, &b)| (a as i64 * b as i64) as i128)
                .sum();
            for m in k + 1..=maxp {
                c.head[k * w + m] = c.head[k * w + m - 1] + prod(m - 1, k);
            }
            for i in 1..=maxp {
                c.tail[k * w + i] = c.tail[k * w + i - 1] + prod(n - i, k);
            }
        }
        c
    }

    /// `C_ij` for an order-`p` predictor.
    #[inline(always)]
    fn c(&self, p: usize, i: usize, j: usize) -> i128 {
        let (i, j) = if i <= j { (i, j) } else { (j, i) };
        let k = j - i;
        self.lag[k] - self.head[k * self.w + p - i] - self.tail[k * self.w + i]
    }

    /// Residual energy of the quantized predictor `qlp`, `shift` (the
    /// decoder's floor, under 1 per sample, is ignored).
    fn energy(&self, qlp: &[i32], shift: i32) -> f64 {
        let p = qlp.len();
        let mut a = [0i64; WIDE_MAX_ORDER + 1];
        a[0] = 1i64 << shift;
        for (ai, &q) in a[1..].iter_mut().zip(qlp) {
            *ai = -(q as i64);
        }
        let a = &a[..=p];
        let mut e = 0i128;
        for (i, &ai) in a.iter().enumerate() {
            let row: i128 = a
                .iter()
                .enumerate()
                .map(|(j, &aj)| aj as i128 * self.c(p, i, j))
                .sum();
            e += ai as i128 * row;
        }
        // |a| <= 2^15 and |C| <= 2^74: at most 13² terms of 2^104, far inside
        // i128. A sum of squares, so never negative.
        e as f64 * pow2(-2 * shift)
    }
}

/// At most this many of the best energy-ranked LPC candidates get their
/// exact residual computed, and only those within [`WIDE_REALIZE_MARGIN`] of
/// the best estimate. On a 384 kHz / 32-bit track 1 / 2 / 4 gave 326.31 /
/// 326.23 / 326.21 MB; realizing every candidate is several times slower.
const WIDE_REALIZE_MAX: usize = 2;
const WIDE_REALIZE_MARGIN: f64 = 0.01;

/// The best wide LPC subframe and the window it came from (index into
/// [`wide_autocs`]). Every window's predictor at every order
/// `1..=max_order` is quantized and ranked by [`WideCov::energy`], the
/// exact residual energy of the *quantized* predictor. The float Levinson
/// error cannot rank them: on oversampled content the predictors are sharp
/// (coefficients up to ~60), and quantization costs some of them far more
/// than others. The best few get their exact residual, re-ranked by a Rice
/// estimate; those within 2% of the best get partition plans.
fn wide_best_lpc(
    samples: &[i32],
    bps: u32,
    max_order: usize,
    seg_windows: usize,
    wins: &WindowCache,
    stats: &mut EncodeStats,
    scratch: &mut EncodeScratch,
) -> Option<(usize, LpcCandidate)> {
    let n = samples.len();
    let autocs = wide_autocs(samples, max_order, seg_windows, wins, scratch);
    let cov = WideCov::new(samples, max_order);

    struct Cand {
        est: f64,
        widx: usize,
        qlp: Vec<i32>,
        shift: i32,
    }
    let mut cands: Vec<Cand> = Vec::new();
    let mut coeffs = [[0.0f64; WIDE_MAX_ORDER]; WIDE_MAX_ORDER];
    for (widx, autoc) in autocs.iter().enumerate() {
        if autoc[0] <= 0.0 {
            continue;
        }
        let found = levinson_all(&autoc[..=max_order], max_order, &mut coeffs);
        if found < max_order {
            stats.lpc_levinson_exhausted += 1;
        }
        for o in 1..=found {
            let Some((qlp, shift)) = quantize_lpc(&coeffs[o - 1][..o], LPC_PRECISION) else {
                continue;
            };
            let m = (n - o) as f64;
            let e = cov.energy(&qlp, shift);
            let est = wide_lpc_header_bits(o, bps) as f64 + m * (0.5 * math::log2(e / m)).max(0.0);
            cands.push(Cand {
                est,
                widx,
                qlp,
                shift,
            });
        }
    }
    cands.sort_by(|a, b| a.est.total_cmp(&b.est));

    // Realize residuals best-first: up to WIDE_REALIZE_MAX within the margin
    // of the best estimate, and past them (whatever the margin) only until
    // one residual fits 30 bits.
    let best_est = cands.first()?.est;
    let mut resids: Vec<(usize, WideResidual)> = Vec::new();
    for (rank, c) in cands.into_iter().enumerate() {
        let close =
            rank < WIDE_REALIZE_MAX && c.est <= best_est + best_est.abs() * WIDE_REALIZE_MARGIN;
        if !close && !resids.is_empty() {
            break;
        }
        let Some(res) = lpc_residual_wide(samples, &c.qlp, c.shift) else {
            continue; // too wide
        };
        let order = c.qlp.len();
        let abs: u64 = res.iter().map(|&r| r.unsigned_abs() as u64).sum();
        let est = wide_lpc_header_bits(order, bps) + rice_bits_estimate(abs, res.len() as u64);
        resids.push((
            c.widx,
            WideResidual {
                order,
                qlp: c.qlp,
                shift: c.shift,
                res,
                est,
            },
        ));
    }
    let best_rice = resids.iter().map(|(_, r)| r.est).min()?;
    let mut best: Option<(usize, LpcCandidate)> = None;
    for (widx, r) in resids {
        if r.est > best_rice + best_rice / 50 {
            continue;
        }
        let c = r.plan(n, bps, &mut *scratch);
        if best.as_ref().is_none_or(|(_, b)| c.bits < b.bits) {
            best = Some((widx, c));
        }
    }
    best
}

/// Subframe choice for bit depths above 24 (no stereo decorrelation, so every
/// value stays within 32 bits): CONSTANT, LPC ([`wide_best_lpc`]: the
/// level's windows, every order up to its cap), FIXED 0..=4, or VERBATIM.
/// FIXED is realized only when its estimate is within 10% of the best so far,
/// as on the narrow path. LPC or FIXED candidates whose residuals do not fit
/// 30 bits are skipped.
fn analyze_wide(
    arm: &ArmInput<'_>,
    max_lpc_order: usize,
    seg_windows: usize,
    wins: &WindowCache,
    stats: &mut EncodeStats,
    scratch: &mut EncodeScratch,
) -> SubframeChoice {
    let samples: &[i32] = &arm.samples;
    let bps = arm.ebps;
    let wb = arm.wasted as u64;
    let n = samples.len();
    if let Some(&first) = samples.first() {
        if samples.iter().all(|&v| v == first) {
            return SubframeChoice {
                bits: 8 + wb + bps as u64,
                wasted: arm.wasted,
                kind: SubframeKind::Constant(first),
            };
        }
    }
    let mut best = SubframeChoice {
        bits: 8 + wb + n as u64 * bps as u64,
        wasted: arm.wasted,
        kind: SubframeKind::Verbatim,
    };

    // LPC: every (window, order) candidate, ranked by the exact energy of
    // its quantized predictor's residual; only the best few are realized.
    let max_order = max_lpc_order.min(n / 2).min(WIDE_MAX_ORDER);
    if max_order >= 1 {
        debug_assert_eq!(wins.n, n, "window cache not sized for this block");
        if let Some((widx, c)) =
            wide_best_lpc(samples, bps, max_order, seg_windows, wins, stats, scratch)
        {
            if c.bits + wb < best.bits {
                if widx == 1 {
                    stats.lpc_window_second_won += 1;
                }
                best = SubframeChoice {
                    bits: c.bits + wb,
                    wasted: arm.wasted,
                    kind: SubframeKind::Lpc(Box::new(c)),
                };
            }
        }
    }

    // FIXED, cheapest estimate first; a too-wide residual falls through to the
    // next order. Stable sort: ties go to the lower order.
    let sums = fixed_sums_wide(samples);
    let mut orders: Vec<usize> = (0..=4usize.min(n - 1)).collect();
    orders.sort_by_key(|&o| sums[o]);
    for order in orders {
        let est = 8
            + wb
            + order as u64 * bps as u64
            + 6
            + rice_bits_estimate(sums[order], (n - order) as u64);
        if est > best.bits.saturating_add(best.bits / 10) {
            break; // ascending estimates: no later order can win either
        }
        let Some(res) = fixed_residual_wide(samples, order) else {
            continue;
        };
        let plan = plan_partitions(&res, n, order, scratch);
        let bits = 8 + wb + order as u64 * bps as u64 + 6 + plan.bits;
        if bits < best.bits {
            best = SubframeChoice {
                bits,
                wasted: arm.wasted,
                kind: SubframeKind::Fixed { order, res, plan },
            };
        }
        break;
    }
    best
}

/// Choose the cheapest subframe type (CONSTANT / LPC / FIXED / VERBATIM) —
/// the mono / multichannel path (stereo goes through [`decide_stereo`]'s
/// two-phase arm gating instead).
fn analyze_subframe(
    arm: &ArmInput<'_>,
    max_lpc_order: usize,
    wins: &WindowCache,
    stats: &mut EncodeStats,
    scratch: &mut EncodeScratch,
) -> SubframeChoice {
    let est = estimate_arm(arm, max_lpc_order, wins, stats, &mut *scratch);
    realize_arm(arm, est, max_lpc_order, wins, stats, scratch)
}

/// Stereo modes whose estimated cost is within this relative margin of the
/// estimated best are realized exactly and compared; the rest are pruned
/// before any residual work.
const STEREO_EST_MARGIN_PCT: u64 = 1;

/// Choose the cheapest of the four FLAC stereo modes for one block.
/// side = L − R (needs bps+1 bits); mid = (L + R) >> 1 (bps).
///
/// Two-phase: every arm (L, R, mid, side) gets a cheap ESTIMATE (window
/// autocorrelations + Levinson + fixed-order sums); only the arms belonging
/// to estimate-competitive modes are REALIZED (residuals, exact Rice plans).
/// The final mode decision uses exact realized costs.
fn decide_stereo<'a>(
    l: &'a [i32],
    r: &'a [i32],
    bps: u32,
    max_lpc_order: usize,
    wins: &WindowCache,
    stats: &mut EncodeStats,
    scratch: &mut EncodeScratch,
) -> (u64, Vec<Subframe<'a>>) {
    // Mid and side in one pass, owned by their arms: a chosen one moves out
    // as the subframe's samples instead of being copied out of a borrow.
    // One fused pass writes both into reserved space (two `extend`s read
    // l and r twice).
    let n = l.len().min(r.len());
    let mut side: Vec<i32> = Vec::with_capacity(n);
    let mut mid: Vec<i32> = Vec::with_capacity(n);
    for (((s, m), &a), &b) in side.spare_capacity_mut()[..n]
        .iter_mut()
        .zip(&mut mid.spare_capacity_mut()[..n])
        .zip(l)
        .zip(r)
    {
        s.write(a - b);
        m.write((a + b) >> 1);
    }
    // SAFETY: the loop above initialized the first `n` slots of both.
    unsafe {
        side.set_len(n);
        mid.set_len(n);
    }

    // Wasted-bits detection + shift per arm, then estimates for all four.
    let arms = [
        ArmInput::prepare(l, bps),
        ArmInput::prepare(r, bps),
        ArmInput::prepare_owned(mid, bps),
        ArmInput::prepare_owned(side, bps + 1),
    ];
    let mut ests = [
        estimate_arm(&arms[0], max_lpc_order, wins, stats, &mut *scratch),
        estimate_arm(&arms[1], max_lpc_order, wins, stats, &mut *scratch),
        estimate_arm(&arms[2], max_lpc_order, wins, stats, &mut *scratch),
        estimate_arm(&arms[3], max_lpc_order, wins, stats, &mut *scratch),
    ];
    // Mode order: independent / left-side / right-side / mid-side.
    let mode_arms: [[usize; 2]; 4] = [[0, 1], [0, 3], [3, 1], [2, 3]];
    let est_costs: Vec<u64> = mode_arms
        .iter()
        .map(|&[a, b]| ests[a].est_bits + ests[b].est_bits)
        .collect();
    let best_est = *est_costs.iter().min().expect("4 modes");
    let cutoff = best_est + best_est * STEREO_EST_MARGIN_PCT / 100;
    let candidate: Vec<bool> = est_costs.iter().map(|&c| c <= cutoff).collect();

    // Phase 2: realize exactly the arms candidate modes need.
    let mut choices: [Option<SubframeChoice>; 4] = [None, None, None, None];
    for (m, &is_cand) in candidate.iter().enumerate() {
        if !is_cand {
            continue;
        }
        for &arm in &mode_arms[m] {
            if choices[arm].is_none() {
                choices[arm] = Some(realize_arm(
                    &arms[arm],
                    core::mem::take(&mut ests[arm]),
                    max_lpc_order,
                    wins,
                    stats,
                    &mut *scratch,
                ));
            }
        }
    }

    // Exact decision over the candidate modes (ties → lowest mode index,
    // matching the original exhaustive search's ordering).
    let mut mode = usize::MAX;
    let mut best_bits = u64::MAX;
    for (m, &is_cand) in candidate.iter().enumerate() {
        if !is_cand {
            continue;
        }
        let [a, b] = mode_arms[m];
        let bits = choices[a].as_ref().expect("realized").bits
            + choices[b].as_ref().expect("realized").bits;
        if bits < best_bits {
            best_bits = bits;
            mode = m;
        }
    }
    debug_assert!(mode < 4);

    // Hand back the two chosen arms: their (possibly wasted-shifted) samples,
    // effective bit depth, and choices.
    let assignment = [1u64, 8, 9, 10][mode];
    let [a, b] = mode_arms[mode];
    let mut arms = arms;
    let mut take_arm = |arm: usize, choices: &mut [Option<SubframeChoice>; 4]| {
        let input = core::mem::replace(
            &mut arms[arm],
            ArmInput {
                samples: alloc::borrow::Cow::Borrowed(&[]),
                ebps: 0,
                wasted: 0,
            },
        );
        let ebps = input.ebps;
        let choice = choices[arm].take().expect("chosen arm realized");
        // A borrow of the held channel stays a borrow; mid/side move out.
        (input.into_cow(), ebps, choice)
    };
    let first = take_arm(a, &mut choices);
    let second = take_arm(b, &mut choices);
    (assignment, vec![first, second])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On content inside the narrow bounds (24-bit), the wide residuals equal
    /// the narrow kernels' exactly: same predictor, same arithmetic result.
    #[test]
    fn wide_residuals_match_narrow_on_24_bit() {
        let mut x = 3u64;
        let s: Vec<i32> = (0..3000)
            .map(|i| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((i as f64 * 0.05).sin() * 8.0e6) as i32 + ((x >> 50) as i32 & 0xFF) - 128
            })
            .collect();
        let peak = s.iter().fold(0, |a, &v| a | v.unsigned_abs());
        for order in 0..=4 {
            assert_eq!(
                fixed_residual_wide(&s, order).unwrap(),
                fixed_residual(&s, order),
                "fixed {order}"
            );
        }
        let sums = fixed_sums_wide(&s);
        assert_eq!(sums, fixed_sums_scalar(&s));
        for order in 1..=LPC_MAX_ORDER {
            for shift in [0, 9, 13] {
                // Coefficients summing to about 1.0 at this shift, so the
                // prediction stays near the signal and residuals fit 30 bits.
                let one = 1i32 << shift;
                let qlp: Vec<i32> = (0..order as i32)
                    .map(|j| one / order as i32 + if j % 2 == 0 { 1 } else { -1 })
                    .collect();
                assert_eq!(
                    lpc_residual_wide(&s, &qlp, shift).unwrap(),
                    lpc_residual_scalar(&s, &qlp, shift, order, peak),
                    "lpc order {order} shift {shift}"
                );
            }
        }
    }

    /// A residual at or beyond 2^30 makes the wide candidate ineligible
    /// instead of wrapping.
    #[test]
    fn wide_residuals_reject_beyond_30_bits() {
        let s = [i32::MAX, i32::MIN, i32::MAX, i32::MIN, 0];
        assert!(fixed_residual_wide(&s, 1).is_none());
        assert!(lpc_residual_wide(&s, &[1], 0).is_none());
        let long: Vec<i32> = (0..40)
            .map(|i| if i % 2 == 0 { i32::MAX } else { i32::MIN })
            .collect();
        assert!(lpc_residual_wide(&long, &[1; 13], 0).is_none());
        let edge = [0, (1 << 30) - 1, 0, -(1 << 30) + 1];
        assert!(fixed_residual_wide(&edge, 0).is_some());
        assert!(fixed_residual_wide(&[0, 1 << 30], 0).is_none());
    }

    /// [`WideCov::energy`] is the quantized predictor's residual energy
    /// exactly: every order, on full-scale 32-bit content and on a block just
    /// long enough for the order, against the sum taken directly in i128.
    #[test]
    fn wide_cov_energy_is_exact() {
        let mut x = 99u64;
        let mut rnd = move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 32) as u32 as i32
        };
        for n in [24usize, 25, 500, BLOCK_SIZE] {
            let mut s: Vec<i32> = (0..n).map(|_| rnd()).collect();
            s[0] = i32::MIN;
            s[n - 1] = i32::MAX;
            let cov = WideCov::new(&s, LPC_MAX_ORDER);
            for p in 1..=LPC_MAX_ORDER {
                // Extreme coefficients: the largest |a| the quantizer emits.
                let qlp: Vec<i32> = (0..p)
                    .map(|j| match j % 3 {
                        0 => (1 << 14) - 1,
                        1 => -(1 << 14),
                        _ => rnd() >> 18,
                    })
                    .collect();
                for shift in [0, 9, 15] {
                    let mut want = 0i128;
                    for t in p..n {
                        let mut r = (s[t] as i128) << shift;
                        for (j, &q) in qlp.iter().enumerate() {
                            r -= q as i128 * s[t - 1 - j] as i128;
                        }
                        want += r * r;
                    }
                    let want = want as f64 * pow2(-2 * shift);
                    assert_eq!(cov.energy(&qlp, shift), want, "n={n} p={p} shift={shift}");
                }
            }
        }
    }

    /// The wide path's Q30 windows: the autocorrelation tracks the float one
    /// on full-scale 32-bit content, for the full-block windows and a piece.
    #[test]
    fn wide_autocorrelation_tracks_float() {
        let n = BLOCK_SIZE;
        let s: Vec<i32> = (0..n)
            .map(|i| (2.1e9 * (i as f64 * 0.003).sin() + 1e5 * (i as f64 * 1.7).cos()) as i32)
            .collect();
        let mut cache = WindowCache {
            wide: true,
            ..WindowCache::default()
        };
        cache.ensure(n, BLOCK_SIZE);
        let set = cache.wide();
        assert_eq!(set.pieces.len(), PIECES.len());
        let mut scratch = EncodeScratch::default();
        let mut check = |seg: &[i32], win: Window<'_, u32>, alpha: f64| {
            let want = autocorr_f64(seg, &tukey_window(seg.len(), alpha), 12);
            autocorrelation(seg, 12, win, &mut scratch);
            for (lag, (&got, &w)) in scratch.autoc.iter().zip(&want).enumerate() {
                let err = (got - w).abs() / want[0];
                // The 25-bit normalisation leaves ~1e-9; Q15 windows ~1e-5.
                assert!(err < 1e-7, "len={} lag={lag}: {got} vs {w}", seg.len());
            }
        };
        for (k, &alpha) in WINDOW_ALPHAS.iter().enumerate() {
            check(&s, set.full[k].view(), alpha);
        }
        let (d, i, j) = PIECES[3];
        check(&s[n * i / d..n * j / d], set.pieces[3].view(), PIECE_ALPHA);
    }

    /// The tapers are the full-length window's non-1.0 samples in Q15, and
    /// every sample between them is exactly 1.0 — at every size a stream can
    /// end on (the short final block), plus the degenerate ones.
    #[test]
    fn tukey_tapers_match_full_window() {
        assert_eq!(q15(1.0), 1 << WIN_Q);
        for n in (0..=600).chain([1023, 1024, 3904, 4000, 4095, BLOCK_SIZE]) {
            for &alpha in &WINDOW_ALPHAS {
                let full = tukey_window(n, alpha);
                let t = tukey_tapers(n, alpha);
                let (h, tl) = (t.head.len(), t.tail.len());
                assert!(h + tl <= n, "n={n} alpha={alpha}");
                let q = |v: &[f64]| v.iter().map(|&x| q15(x)).collect::<Vec<_>>();
                assert_eq!(t.head, q(&full[..h]), "head n={n} alpha={alpha}");
                assert_eq!(t.tail, q(&full[n - tl..]), "tail n={n} alpha={alpha}");
                assert!(
                    full[h..n - tl]
                        .iter()
                        .all(|&w| w.to_bits() == 1.0f64.to_bits()),
                    "middle n={n} alpha={alpha}"
                );
            }
        }
    }

    /// A Rice-parameter scan over a row truncated at the residual's top bit
    /// picks the same parameter and cost as the scan over all 31 sums, for
    /// every kmax the planner uses and every magnitude class.
    #[test]
    fn truncated_rice_row_matches_full_row() {
        let mut x = 5u64;
        for bits in 0..=31u32 {
            for n in [1usize, 2, 7, 64, 1024] {
                let res: Vec<i32> = (0..n)
                    .map(|_| {
                        x = x
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let v = (x >> 32) as u32 as i64;
                        let m = if bits == 0 {
                            0
                        } else {
                            (1i64 << (bits - 1)) - 1
                        };
                        (if m == 0 { 0 } else { v % (m + 1) - m / 2 }) as i32
                    })
                    .chain(core::iter::once(if bits >= 2 {
                        -(1i32 << (bits - 2))
                    } else {
                        0
                    }))
                    .collect();
                let mut full = [0u64; RICE_KMAX + 1];
                rice_sums_scalar_into(&res, &mut full);
                let stride = rice_stride(&res);
                let mut row = vec![0u64; stride];
                rice_sums_into(&res, &mut row);
                assert_eq!(&full[..stride], &row[..], "bits={bits} n={n}");
                assert!(full[stride..].iter().all(|&s| s == 0) || stride == RICE_KMAX + 1);
                for cnt in [res.len() as u64, res.len() as u64 + 3] {
                    for kmax in [RICE_KMAX_M0, RICE_KMAX] {
                        assert_eq!(
                            best_k_from_sums(&full, cnt, kmax),
                            best_k_from_sums(&row, cnt, kmax),
                            "bits={bits} n={n} cnt={cnt} kmax={kmax}"
                        );
                    }
                }
            }
        }
    }

    /// A reused encoder produces exactly the streams fresh encoders do, chunk
    /// after chunk, across block-size changes (full, short tail, full again,
    /// a different tail, a stream shorter than one block).
    #[test]
    fn finish_and_reset_matches_fresh_encoders() {
        for &(ch, level) in &[(1u32, 0u32), (1, 8), (2, 5)] {
            let mut reused = Encoder::new(16000, ch, 16).unwrap();
            reused.set_compression_level(level);
            for (k, &n) in [8000usize, 8192, 3904, 8000, 100, 12345, 8000]
                .iter()
                .enumerate()
            {
                let x: Vec<i32> = (0..n * ch as usize)
                    .map(|i| (((i * 7919 + k * 104729) % 2003) as i32 - 1001) * 13)
                    .collect();
                let mut fresh = Encoder::new(16000, ch, 16).unwrap();
                fresh.set_compression_level(level);
                fresh.push_interleaved(&x).unwrap();
                let want = fresh.finish();
                reused.push_interleaved(&x).unwrap();
                let builds = reused.wins.builds;
                assert_eq!(
                    reused.finish_and_reset(),
                    want,
                    "ch={ch} level={level} chunk={k} n={n}"
                );
                // Chunk 3 (8000 = 4096 + a 3904 tail) follows chunk 2 (3904):
                // its tail's tapers must come from the cache.
                if k == 3 {
                    assert_eq!(reused.wins.builds, builds, "tail tapers rebuilt");
                }
            }
        }
    }

    /// A set block size frames the stream at that size — STREAMINFO and
    /// every frame but a short last one — round-trips, and is kept by
    /// `finish_and_reset`; sizes FLAC cannot express are refused.
    #[test]
    fn set_block_size_frames_the_stream() {
        let mut enc = Encoder::new(44100, 2, 16).unwrap();
        for bad in [0, 15, 65536] {
            assert_eq!(enc.set_block_size(bad), Err(EncodeError::BadBlockSize(bad)));
        }
        enc.set_compression_level(8);
        for bs in [16u32, 1152, 1176, 4608, 65535] {
            enc.set_block_size(bs).unwrap();
            for n in [3 * bs as usize, 3 * bs as usize + 5] {
                let x: Vec<i32> = (0..n * 2)
                    .map(|i| (((i * 7919) % 2003) as i32 - 1001) * 13)
                    .collect();
                enc.push_interleaved(&x).unwrap();
                let stream = enc.finish_and_reset();
                let mut dec = crate::decode::Decoder::new(&stream).unwrap();
                let info = dec.streaminfo();
                assert_eq!((info.min_block_size, info.max_block_size), (bs, bs));
                let mut out = vec![Vec::new(), Vec::new()];
                let mut sizes = Vec::new();
                while let Some(size) = dec.next_frame(&mut out).unwrap() {
                    sizes.push(size);
                }
                let tail = n % bs as usize;
                let mut want = vec![bs as usize; n / bs as usize];
                if tail != 0 {
                    want.push(tail);
                }
                assert_eq!(sizes, want, "bs={bs} n={n}");
                let left: Vec<i32> = x.iter().step_by(2).copied().collect();
                let right: Vec<i32> = x.iter().skip(1).step_by(2).copied().collect();
                assert_eq!(out, [left, right], "bs={bs} n={n}");
                // The same stream again: nominal and tail windows both cached.
                enc.push_interleaved(&x).unwrap();
                let builds = enc.wins.builds;
                assert_eq!(enc.finish_and_reset(), stream, "bs={bs} n={n}");
                assert_eq!(enc.wins.builds, builds, "bs={bs} n={n}: windows rebuilt");
            }
        }
    }

    /// The i32 fixed-order sums equal the original i64 formulas exactly, at
    /// full-scale 25-bit (side-channel) amplitude and every tail length.
    #[test]
    fn fixed_sums_i32_match_i64_reference() {
        let reference = |x: &[i32]| -> [u64; 5] {
            let mut sums = [0u64; 5];
            sums[0] = x.iter().map(|&v| (v as i64).unsigned_abs()).sum();
            for i in 1..x.len() {
                let s = |j: usize| if i >= j { x[i - j] as i64 } else { 0 };
                sums[1] += (s(0) - s(1)).unsigned_abs();
                if i >= 2 {
                    sums[2] += (s(0) - 2 * s(1) + s(2)).unsigned_abs();
                }
                if i >= 3 {
                    sums[3] += (s(0) - 3 * s(1) + 3 * s(2) - s(3)).unsigned_abs();
                }
                if i >= 4 {
                    sums[4] += (s(0) - 4 * s(1) + 6 * s(2) - 4 * s(3) + s(4)).unsigned_abs();
                }
            }
            sums
        };
        let m = (1i32 << 24) - 1;
        let mut x = 9u64;
        for n in [1usize, 2, 3, 4, 5, 7, 8, 9, 64, 4095, 4096] {
            for pattern in 0..3 {
                let v: Vec<i32> = (0..n)
                    .map(|i| match pattern {
                        0 => {
                            if i % 2 == 0 {
                                m
                            } else {
                                -m - 1
                            }
                        }
                        1 => {
                            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                            ((x >> 39) as i32) - (1 << 24)
                        }
                        _ => ((i as f64 * 0.3).sin() * 30000.0) as i32,
                    })
                    .collect();
                assert_eq!(
                    fixed_sums_scalar(&v),
                    reference(&v),
                    "n={n} pattern={pattern}"
                );
            }
        }
    }

    /// The u32-accumulated Rice sums equal the u64 ones wherever the u32 path
    /// is taken (and the row is exact at the bound: top bit + count bits = 32).
    #[test]
    fn rice_sums_u32_path_is_exact() {
        let mut x = 21u64;
        for bits in [1u32, 8, 16, 18, 20, 26] {
            for n in [1usize, 16, 64, 4096] {
                let res: Vec<i32> = (0..n)
                    .map(|_| {
                        x = x.wrapping_mul(6364136223846793005).wrapping_add(3);
                        let lim = 1i64 << (bits - 1);
                        (((x >> 20) as i64 % (2 * lim)) - lim) as i32
                    })
                    .collect();
                let stride = rice_stride(&res);
                let mut fast = vec![0u64; stride];
                rice_sums_scalar_into(&res, &mut fast);
                let mut full = [0u64; RICE_KMAX + 1];
                rice_sums_scalar_into(&res, &mut full); // u64 path
                assert_eq!(&full[..stride], &fast[..], "bits={bits} n={n}");
            }
        }
    }

    /// The 32-bit LPC residual path equals the 64-bit one wherever its bound
    /// admits it — including right at the bound — and the bound refuses the
    /// cases that would overflow.
    #[test]
    fn lpc_residual_i32_path_matches_i64() {
        // The i64 reference, or None when a residual would not fit i32 (a
        // predictor the encoder never produces; the i64 path overflows there
        // too).
        let reference = |s: &[i32], q: &[i32], shift: i32| -> Option<Vec<i32>> {
            (q.len()..s.len())
                .map(|i| {
                    let sum: i64 = (0..q.len())
                        .map(|j| q[j] as i64 * s[i - 1 - j] as i64)
                        .sum();
                    i32::try_from(s[i] as i64 - (sum >> shift)).ok()
                })
                .collect()
        };
        let mut x = 5u64;
        let mut rnd = move |m: i64| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 20) as i64 % (2 * m + 1)) - m
        };
        let (mut checked, mut on_i32_path) = (0u32, 0u32);
        for order in 1..=14usize {
            for &(amp, cmax) in &[
                (300i64, 9000i64),
                (32767, 16383),
                (32767, 1 << 14),
                ((1 << 24) - 1, 16383),
            ] {
                let s: Vec<i32> = (0..777).map(|_| rnd(amp) as i32).collect();
                let q: Vec<i32> = (0..order).map(|_| rnd(cmax) as i32).collect();
                for shift in [0, 9, 14, 15] {
                    let Some(want) = reference(&s, &q, shift) else {
                        continue;
                    };
                    let big = s.iter().fold(0u32, |a, &v| a | v.unsigned_abs());
                    let sum_abs: u64 = q.iter().map(|&c| c.unsigned_abs() as u64).sum();
                    on_i32_path += (sum_abs << (32 - big.leading_zeros()) < 1u64 << 31) as u32;
                    checked += 1;
                    assert_eq!(
                        lpc_residual_scalar(&s, &q, shift, order, big),
                        want,
                        "order={order} amp={amp} cmax={cmax} shift={shift}"
                    );
                }
            }
        }
        // Both paths were exercised.
        assert!(
            on_i32_path > 20 && checked - on_i32_path > 20,
            "{on_i32_path} of {checked} on the i32 path"
        );
    }

    /// The paired-lag scalar kernel equals the per-lag definition for every
    /// lag count (odd counts take the single-lag tail) and short blocks.
    #[test]
    fn autocorr_int_scalar_matches_definition() {
        let mut x = 7u64;
        for n in [2usize, 3, 13, 14, 100, 4096] {
            let w: Vec<u64> = (0..n)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(11);
                    (((x >> 32) as i32) >> 7) as i64 as u64
                })
                .collect();
            for lags in 1..=13.min(n) {
                let want: Vec<i64> = (0..lags)
                    .map(|l| {
                        (0..n - l)
                            .map(|i| w[i] as i32 as i64 * w[i + l] as i32 as i64)
                            .sum()
                    })
                    .collect();
                let mut got = vec![0i64; lags];
                autocorr_int_scalar(&w, &mut got);
                assert_eq!(got, want, "n={n} lags={lags}");
            }
        }
    }

    /// The generated `BLOCK_SIZE` table is `tukey_tapers`, value for value — on
    /// `libm` builds by construction (it is generated with `libm::cos`), and on
    /// platform-libm builds because a last-bit difference in `cos` does not
    /// move a Q15 rounding (checked here on every host the suite runs on).
    /// `RUSTY_FLAC_REGEN_WINDOWS=1` (with `--features libm`) rewrites
    /// `src/window_table.rs`.
    #[test]
    fn window_table_is_runtime_tukey() {
        let tapers = WINDOW_ALPHAS.map(|a| tukey_tapers(BLOCK_SIZE, a));
        #[cfg(feature = "libm")]
        if std::env::var_os("RUSTY_FLAC_REGEN_WINDOWS").is_some() {
            use std::fmt::Write as _;
            let mut src = std::string::String::from(
                "//! Tukey tapers of the two LPC windows for a full block (4096 samples,\n\
                 //! alpha 0.5 and 0.2) in Q15. GENERATED with `libm::cos` by\n\
                 //! `RUSTY_FLAC_REGEN_WINDOWS=1 cargo test --features libm --lib\n\
                 //! window_table_is_runtime_tukey`, which also gates this file against the\n\
                 //! runtime computation. Do not edit.\n",
            );
            for (k, t) in tapers.iter().enumerate() {
                for (name, v) in [("HEAD", &t.head), ("TAIL", &t.tail)] {
                    writeln!(
                        src,
                        "\npub(crate) static {name}_{k}: [u16; {}] = [",
                        v.len()
                    )
                    .unwrap();
                    for chunk in v.chunks(12) {
                        src.push_str("   ");
                        for x in chunk {
                            write!(src, " {x},").unwrap();
                        }
                        src.push('\n');
                    }
                    src.push_str("];\n");
                }
            }
            let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/window_table.rs");
            std::fs::write(path, src).unwrap();
            return;
        }
        for (k, t) in tapers.iter().enumerate() {
            let w = full_block_window(k);
            assert_eq!(w.head, &t.head[..], "head {k}");
            assert_eq!(w.tail, &t.tail[..], "tail {k}");
        }
    }

    /// The integer autocorrelation tracks the float one it replaced: same
    /// units, relative error far below anything LPC order selection or the
    /// Levinson recursion can see — on quiet, loud, tonal and full-scale
    /// 24-bit content, full and short blocks.
    #[test]
    fn int_autocorrelation_tracks_float() {
        let mut x = 17u64;
        let mut rnd = move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for &(n, amp, tone) in &[
            (BLOCK_SIZE, 300.0, 0.0),
            (BLOCK_SIZE, 30000.0, 0.3),
            (BLOCK_SIZE, 8_000_000.0, 1.0),
            (BLOCK_SIZE, 16_700_000.0, 0.0),
            (3904, 2000.0, 0.5),
            (37, 5.0, 0.0),
        ] {
            let s: Vec<i32> = (0..n)
                .map(|i| (amp * (tone * (i as f64 * 0.02).sin() + (1.0 - tone) * rnd())) as i32)
                .collect();
            let mut cache = WindowCache::default();
            cache.ensure(n, BLOCK_SIZE);
            for (k, &alpha) in WINDOW_ALPHAS.iter().enumerate() {
                let want = autocorr_f64(&s, &tukey_window(n, alpha), 12);
                let mut scratch = EncodeScratch::default();
                autocorrelation(&s, 12, cache.get(k), &mut scratch);
                for (lag, (&got, &w)) in scratch.autoc.iter().zip(&want).enumerate() {
                    let err = (got - w).abs() / want[0];
                    assert!(err < 2e-4, "n={n} amp={amp} k={k} lag={lag}: {got} vs {w}");
                }
            }
        }
    }

    /// The normalisation keeps every lag sum inside i64: full-scale 25-bit
    /// (side-channel) content, the largest block, every lag — checked against
    /// the same sums in i128.
    #[test]
    fn int_autocorrelation_cannot_overflow() {
        let n = BLOCK_SIZE;
        for pattern in 0..3 {
            let s: Vec<i32> = (0..n)
                .map(|i| {
                    let m = (1i32 << 24) - 1;
                    match pattern {
                        0 => m,
                        1 => {
                            if i % 2 == 0 {
                                m
                            } else {
                                -m - 1
                            }
                        }
                        _ => -m - 1,
                    }
                })
                .collect();
            let mut cache = WindowCache::default();
            cache.ensure(n, BLOCK_SIZE);
            let mut scratch = EncodeScratch::default();
            autocorrelation(&s, 32, cache.get(0), &mut scratch);
            let w = &scratch.words;
            for lag in 0..=32 {
                let wide: i128 = (0..n - lag)
                    .map(|i| (w[i] as i32 as i128) * (w[i + lag] as i32 as i128))
                    .sum();
                assert!(wide.abs() < (1i128 << 62), "pattern {pattern} lag {lag}");
                let mut out = [0i64; 33];
                autocorr_int_scalar(w, &mut out);
                assert_eq!(out[lag] as i128, wide, "pattern {pattern} lag {lag}");
            }
        }
    }

    fn sine_stereo(n: usize) -> (Vec<i32>, Vec<i32>) {
        let l: Vec<i32> = (0..n)
            .map(|i| ((i as f64 * 0.05).sin() * 20000.0) as i32)
            .collect();
        let r = vec![1234i32; n];
        (l, r)
    }

    fn decode_with_claxon(stream: &[u8]) -> (u32, u32, u32, Vec<Vec<i32>>) {
        let mut reader = claxon::FlacReader::new(std::io::Cursor::new(stream)).expect("parse");
        let info = reader.streaminfo();
        let ch = info.channels as usize;
        let mut chans = vec![Vec::new(); ch];
        let mut c = 0usize;
        for s in reader.samples() {
            chans[c].push(s.expect("sample"));
            c = (c + 1) % ch;
        }
        (info.sample_rate, info.channels, info.bits_per_sample, chans)
    }

    #[test]
    fn roundtrip_lossless_stereo_s16() {
        let (l, r) = sine_stereo(10_000);
        let mut enc = Encoder::new(44100, 2, 16).unwrap();
        enc.push_planar(&[&l, &r]).unwrap();
        let stream = enc.finish();
        assert_eq!(&stream[..4], b"fLaC");
        let (sr, ch, bps, chans) = decode_with_claxon(&stream);
        assert_eq!((sr, ch, bps), (44100, 2, 16));
        assert_eq!(chans[0], l);
        assert_eq!(chans[1], r);
        // It must actually compress (sine + constant).
        assert!(
            stream.len() < 10_000 * 4 / 2,
            "no compression: {}",
            stream.len()
        );
    }

    #[test]
    fn roundtrip_interleaved_matches_planar() {
        let (l, r) = sine_stereo(5_000);
        let inter: Vec<i32> = l.iter().zip(&r).flat_map(|(&a, &b)| [a, b]).collect();

        let mut e1 = Encoder::new(48000, 2, 16).unwrap();
        e1.push_planar(&[&l, &r]).unwrap();
        let mut e2 = Encoder::new(48000, 2, 16).unwrap();
        e2.push_interleaved(&inter).unwrap();
        assert_eq!(e1.finish(), e2.finish());
    }

    #[test]
    fn compression_level_lossless_and_monotonic() {
        // Noisy-ish deterministic signal so LPC order matters.
        let n = 20_000;
        let mut x = 0i64;
        let s: Vec<i32> = (0..n)
            .map(|i| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((x >> 33) & 0xFF) as i32 - 128;
                ((i as f64 * 0.03).sin() * 12000.0) as i32 + noise
            })
            .collect();
        let encode_at = |level: u32| -> Vec<u8> {
            let mut e = Encoder::new(44100, 1, 16).unwrap();
            e.set_compression_level(level);
            e.push_planar(&[&s]).unwrap();
            e.finish()
        };
        let l0 = encode_at(0);
        let l8 = encode_at(8);
        for stream in [&l0, &l8] {
            let (_, _, _, chans) = decode_with_claxon(stream);
            assert_eq!(chans[0], s, "compression-level round-trip is not lossless");
        }
        assert!(l8.len() <= l0.len(), "level 8 larger than level 0");
    }

    #[test]
    fn stats_paths_wired() {
        let (l, r) = sine_stereo(10_000);
        let mut enc = Encoder::new(44100, 2, 16).unwrap();
        enc.push_planar(&[&l, &r]).unwrap();
        let (_, stats) = enc.finish_with_stats();
        assert!(stats.frames > 0);
        assert!(stats.sub_constant > 0, "constant channel not detected");
        assert!(
            stats.sub_lpc + stats.sub_fixed > 0,
            "no predictive subframes"
        );
    }

    /// The AVX2 integer lag sums equal the scalar twin's exactly (integer
    /// sums: any order is the same sum), on every length and tail.
    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    fn autocorr_int_avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut x = 3u64;
        for n in [15usize, 64, 1000, 4096, 4097] {
            let w: Vec<u64> = (0..n)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(99);
                    (((x >> 32) as i32) >> 7) as i64 as u64 // |v| < 2^25
                })
                .collect();
            let mut a = [0i64; 13];
            let mut b = [0i64; 13];
            autocorr_int_scalar(&w, &mut a);
            unsafe { autocorr_int_avx2(&w, &mut b) };
            assert_eq!(a, b, "n={n}");
        }
    }

    /// 16-bit content stored in a 24-bit container (8 zero LSBs per sample)
    /// must trigger the wasted-bits path: dramatically smaller than the naive
    /// coding, still exactly lossless.
    #[test]
    fn wasted_bits_on_16_in_24_content() {
        let n = 20_000;
        let mut x = 5u64;
        let s16: Vec<i32> = (0..n)
            .map(|i| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(11);
                ((i as f64 * 0.02).sin() * 9000.0) as i32 + ((x >> 40) & 0xFF) as i32 - 128
            })
            .collect();
        let s24: Vec<i32> = s16.iter().map(|&v| v << 8).collect();

        let encode = |data: &Vec<i32>, bps: u32| -> Vec<u8> {
            let mut e = Encoder::new(48000, 1, bps).unwrap();
            e.push_planar(&[data]).unwrap();
            e.finish()
        };
        let native16 = encode(&s16, 16);
        let in24 = encode(&s24, 24);

        // Lossless round-trip of the 24-bit stream.
        let (info, chans) = crate::decode::decode(&in24).unwrap();
        assert_eq!(info.bits_per_sample, 24);
        assert_eq!(chans[0], s24, "wasted-bits round-trip broke losslessness");

        // The 24-bit container must cost within ~2% of the true 16-bit coding
        // (8 zero LSBs are shifted out, not Rice-coded).
        let ratio = in24.len() as f64 / native16.len() as f64;
        assert!(
            ratio < 1.02,
            "wasted-bits not engaging: 24-bit container {} B vs 16-bit {} B ({ratio:.3}x)",
            in24.len(),
            native16.len()
        );

        // And the stats counter must show the path fired.
        let mut e = Encoder::new(48000, 1, 24).unwrap();
        e.push_planar(&[&s24]).unwrap();
        let (_, stats) = e.finish_with_stats();
        assert!(stats.sub_wasted_bits > 0, "wasted-bits counter never fired");
    }

    /// The AVX2 shifted-sum kernel is integer math — it must match the scalar
    /// twin EXACTLY on every length (including the empty/short tails).
    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    fn rice_sums_avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut x = 11u64;
        for n in [0usize, 1, 3, 16, 255, 4096] {
            let res: Vec<i32> = (0..n)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(7);
                    ((x >> 30) as i32) >> ((x >> 60) & 15) // wide dynamic range
                })
                .collect();
            let mut scalar = [0u64; RICE_KMAX + 1];
            rice_sums_scalar_into(&res, &mut scalar);
            assert_eq!(scalar, unsafe { rice_sums_avx2(&res) }, "n={n}");
        }
        // The u32-lane kernel: values just under 2^16, up to 65535 of them
        // (the largest sum its guard admits), every row width up to 16.
        for n in [0usize, 1, 7, 32, 4096, 65535] {
            let res: Vec<i32> = (0..n)
                .map(|i| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(7);
                    let u = if i % 3 == 0 {
                        0xffff
                    } else {
                        ((x >> 40) as u32) & 0xffff
                    };
                    // un-zigzag so zigzag(v) == u
                    ((u >> 1) as i32) ^ -((u & 1) as i32)
                })
                .collect();
            for width in 1..=16usize {
                let mut scalar = vec![0u64; width];
                let mut full = [0u64; RICE_KMAX + 1];
                rice_sums_scalar_into(&res, &mut full);
                scalar.copy_from_slice(&full[..width]);
                let mut fast = vec![0u64; width];
                unsafe { rice_sums_avx2_u32(&res, &mut fast) };
                assert_eq!(scalar, fast, "u32 n={n} width={width}");
            }
        }
    }

    /// The AVX2 fixed-order |residual| sums are integer math — exact match
    /// against the scalar twin on every length and alignment.
    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    fn fixed_sums_avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut x = 17u64;
        // Each depth's full-scale domain, including alternating extremes --
        // the worst case for the u32 lane sums between widenings.
        for bps in [16u32, 17, 24, 25] {
            let top = 1i32 << (bps - 1);
            for n in [16usize, 17, 23, 64, 4095, 4096] {
                let s: Vec<i32> = (0..n)
                    .map(|i| {
                        x = x.wrapping_mul(6364136223846793005).wrapping_add(3);
                        if x >> 62 == 0 {
                            if i % 2 == 0 {
                                top - 1
                            } else {
                                -top
                            }
                        } else {
                            (((x >> 33) as i32) >> ((x >> 59) & 7)).clamp(-top, top - 1)
                        }
                    })
                    .collect();
                assert_eq!(
                    fixed_sums_scalar(&s),
                    unsafe { fixed_sums_avx2(&s, bps) },
                    "bps={bps} n={n}"
                );
            }
        }
    }

    /// The FMA-f64 LPC residual must equal the scalar i64 path exactly on
    /// realistic magnitudes (the dispatcher's range guard keeps it off the
    /// degenerate ones).
    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    fn lpc_residual_avx2_matches_scalar() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        let mut x = 23u64;
        let mut i32_checked = 0usize;
        let samples: Vec<i32> = (0..5000)
            .map(|i| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(13);
                ((i as f64 * 0.07).sin() * 3_000_000.0) as i32 + ((x >> 40) & 0xFFF) as i32
            })
            .collect();
        for order in [1usize, 2, 4, 8, 12] {
            let qlp: Vec<i32> = (0..order)
                .map(|j| ((x >> (j * 3)) & 0xFFF) as i32 - 2048)
                .collect();
            for shift in [11i32, 14] {
                let peak = samples.iter().fold(0, |a: u32, &s| a | s.unsigned_abs());
                let a = lpc_residual_scalar(&samples, &qlp, shift, order, peak);
                let b = unsafe { lpc_residual_avx2::<false>(&samples, &qlp, shift, order) };
                let c = unsafe { lpc_residual_avx2::<true>(&samples, &qlp, shift, order) };
                assert_eq!(a, c, "checked order={order} shift={shift}");
                let sum_abs: u64 = qlp.iter().map(|&c| c.unsigned_abs() as u64).sum();
                if sum_abs << (32 - peak.leading_zeros()) < 1u64 << 31 {
                    let d = unsafe { lpc_residual_avx2_i32(&samples, &qlp, shift, order) };
                    assert_eq!(a, d, "i32 order={order} shift={shift}");
                    i32_checked += 1;
                }
                assert_eq!(a, b, "order={order} shift={shift}");
            }
        }
        // Predictions OUTSIDE i32: full-scale 25-bit samples (a 24-bit side
        // channel) against large 15-bit coefficients and a small shift, so
        // some vectors overflow i32 and take the per-vector scalar fallback
        // (the scalar form wraps; the conversion would saturate).
        let wide: Vec<i32> = (0..3000)
            .map(|i| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(13);
                let v = ((i as f64 * 0.013).sin() * 16_000_000.0) as i32;
                if (x >> 33) & 7 == 0 {
                    -v
                } else {
                    v
                }
            })
            .collect();
        // The i32-lane kernel on 16-bit-scale content, where its bound holds.
        let small: Vec<i32> = samples.iter().map(|&v| v >> 7).collect();
        let speak = small.iter().fold(0, |a: u32, &v| a | v.unsigned_abs());
        for order in [1usize, 2, 3, 4, 7, 8, 12] {
            let qlp: Vec<i32> = (0..order)
                .map(|j| ((x >> (j * 3)) & 0x3FF) as i32 - 512)
                .collect();
            for shift in [0i32, 9, 12] {
                let sum_abs: u64 = qlp.iter().map(|&c| c.unsigned_abs() as u64).sum();
                assert!(
                    sum_abs << (32 - speak.leading_zeros()) < 1u64 << 31,
                    "test setup"
                );
                let a = lpc_residual_scalar(&small, &qlp, shift, order, speak);
                let d = unsafe { lpc_residual_avx2_i32(&small, &qlp, shift, order) };
                assert_eq!(a, d, "i32 order={order} shift={shift}");
                i32_checked += 1;
            }
        }
        assert!(i32_checked > 0);
        let mut overflowed = false;
        for order in [2usize, 7, 12] {
            let qlp: Vec<i32> = (0..order)
                .map(|j| if j % 2 == 0 { 16383 } else { -9000 })
                .collect();
            for shift in [0i32, 3, 9] {
                let peak = wide.iter().fold(0, |a: u32, &s| a | s.unsigned_abs());
                let a = lpc_residual_scalar(&wide, &qlp, shift, order, peak);
                let b = unsafe { lpc_residual_avx2::<true>(&wide, &qlp, shift, order) };
                assert_eq!(a, b, "wide order={order} shift={shift}");
                overflowed |= (order..wide.len()).any(|i| {
                    let s: i64 = (0..order)
                        .map(|k| qlp[k] as i64 * wide[i - 1 - k] as i64)
                        .sum();
                    (s >> shift) != ((s >> shift) as i32) as i64
                });
            }
        }
        assert!(overflowed, "test setup: no prediction left i32");
    }

    /// The warm-started search equals the ascending scan on real shifted
    /// sums: every magnitude class, row widths cut short of `kmax`, both
    /// method caps, empty and all-zero partitions.
    #[test]
    fn best_k_matches_ascending_scan() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        for case in 0..4000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let cnt = [0usize, 1, 2, 7, 16, 32, 64, 4096][case % 8];
            let bits = (x >> 58) as u32; // 0..=63 -> magnitude class
            let res: Vec<i32> = (0..cnt)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    if bits % 9 == 0 {
                        0
                    } else {
                        ((x >> 32) as i32) >> (31 - (bits % 26))
                    }
                })
                .collect();
            let mut full = [0u64; RICE_KMAX + 1];
            rice_sums_scalar_into(&res, &mut full);
            let width = 1 + (x as usize >> 3) % (RICE_KMAX + 1);
            let row = &full[..width];
            for kmax in [RICE_KMAX, RICE_KMAX_M0, 0, 3] {
                assert_eq!(
                    best_k_from_sums(row, cnt as u64, kmax),
                    best_k_scan(row, cnt as u64, kmax),
                    "case {case} kmax {kmax} width {width}"
                );
            }
        }
    }

    #[test]
    fn rejects_bad_config() {
        assert!(Encoder::new(44100, 0, 16).is_err());
        assert!(Encoder::new(44100, 9, 16).is_err());
        assert!(Encoder::new(44100, 2, 12).is_err());
        assert!(Encoder::new(0, 2, 16).is_err());
        let mut e = Encoder::new(44100, 2, 16).unwrap();
        assert!(e.push_interleaved(&[1, 2, 3]).is_err());
    }
}
