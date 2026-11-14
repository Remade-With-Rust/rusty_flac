//! Streaming encode (`take_frames` / `finish_frames`): frame by frame, in
//! constant memory, byte-identical to the whole-stream `finish`.

use rusty_flac::{decode, EncodedFrame, Encoder, StreamInfo};

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

/// A tone plus noise at `bps`, interleaved.
fn signal(frames: usize, channels: usize, bps: u32, seed: u64) -> Vec<i32> {
    let mut s = seed;
    let amp = ((1i64 << (bps - 1)) - 1) as f64;
    let mut out = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        for c in 0..channels {
            let tone = ((i as f64) * 0.013 * (c + 1) as f64).sin() * 0.6;
            let noise = (lcg(&mut s) % 2001) as f64 / 1000.0 - 1.0;
            let v = (tone + noise * 0.05) * amp;
            out.push(v.round().clamp(-amp - 1.0, amp) as i32);
        }
    }
    out
}

fn whole(rate: u32, ch: usize, bps: u32, level: u32, bs: u32, pcm: &[i32]) -> Vec<u8> {
    let mut e = Encoder::new(rate, ch as u32, bps).unwrap();
    e.set_compression_level(level);
    e.set_block_size(bs).unwrap();
    e.push_interleaved(pcm).unwrap();
    e.finish()
}

/// Stream `pcm` in pseudo-random pushes, taking frames after each.
fn streamed(
    rate: u32,
    ch: usize,
    bps: u32,
    level: u32,
    bs: u32,
    pcm: &[i32],
    seed: u64,
) -> (Vec<EncodedFrame>, StreamInfo) {
    let mut e = Encoder::new(rate, ch as u32, bps).unwrap();
    e.set_compression_level(level);
    e.set_block_size(bs).unwrap();
    let mut s = seed;
    let mut frames = Vec::new();
    let mut at = 0;
    let n = pcm.len() / ch;
    while at < n {
        let take = (1 + lcg(&mut s) as usize % (3 * bs as usize)).min(n - at);
        e.push_interleaved(&pcm[at * ch..(at + take) * ch]).unwrap();
        frames.extend(e.take_frames());
        at += take;
    }
    let (tail, info) = e.finish_frames();
    frames.extend(tail);
    (frames, info)
}

#[test]
fn streaming_is_byte_identical_to_finish() {
    // (channels, bps, level, block size)
    let shapes = [
        (1usize, 16u32, 5u32, 4096u32),
        (2, 16, 5, 4096),
        (2, 16, 0, 4096),
        (2, 24, 8, 4096),
        (6, 24, 5, 4096),
        (2, 32, 5, 4096),
        (1, 8, 5, 4096),
        (2, 16, 5, 1152),
    ];
    for (ch, bps, level, bs) in shapes {
        let b = bs as usize;
        for frames in [0usize, 5, b - 1, b, b + 1, 3 * b + b / 2] {
            let pcm = signal(frames, ch, bps, 17 + frames as u64);
            let want = whole(48_000, ch, bps, level, bs, &pcm);
            let (got, info) = streamed(48_000, ch, bps, level, bs, &pcm, 99 + frames as u64);
            let mut stream = info.stream_header();
            for f in &got {
                stream.extend_from_slice(&f.data);
            }
            assert_eq!(
                stream, want,
                "{ch} ch {bps}-bit level {level} bs {bs}, {frames} frames"
            );
            // Frame bookkeeping: contiguous, block-sized but the last.
            let mut next = 0u64;
            for (k, f) in got.iter().enumerate() {
                assert_eq!(f.first_sample, next);
                if k + 1 < got.len() {
                    assert_eq!(f.samples, bs);
                }
                next += u64::from(f.samples);
            }
            assert_eq!(next, frames as u64);
            assert_eq!(info.total_samples, frames as u64);
        }
    }
}

#[test]
fn final_streaminfo_matches_what_the_decoder_reads() {
    let pcm = signal(20_000, 2, 24, 5);
    let (frames, info) = streamed(96_000, 2, 24, 8, 4096, &pcm, 3);
    let mut stream = info.stream_header();
    frames
        .iter()
        .for_each(|f| stream.extend_from_slice(&f.data));
    let (read, planes) = decode(&stream).unwrap();
    assert_eq!(read.to_bytes(), info.to_bytes());
    assert_eq!(read.total_samples, 20_000);
    assert_eq!(
        (read.min_frame_size, read.max_frame_size),
        (info.min_frame_size, info.max_frame_size)
    );
    assert!(info.min_frame_size > 0 && info.min_frame_size <= info.max_frame_size);
    let back: Vec<i32> = (0..20_000)
        .flat_map(|i| [planes[0][i], planes[1][i]])
        .collect();
    assert_eq!(back, pcm, "lossless");
}

#[test]
fn a_placeholder_header_then_frames_is_a_valid_stream() {
    // A writer that cannot seek back keeps the placeholder: totals unknown.
    let pcm = signal(9000, 1, 16, 8);
    let mut e = Encoder::new(44_100, 1, 16).unwrap();
    let mut stream = e.stream_info_placeholder().stream_header();
    e.push_interleaved(&pcm).unwrap();
    e.take_frames()
        .iter()
        .for_each(|f| stream.extend_from_slice(&f.data));
    let (frames, info) = e.finish_frames();
    frames
        .iter()
        .for_each(|f| stream.extend_from_slice(&f.data));
    let (read, planes) = decode(&stream).unwrap();
    assert_eq!(read.total_samples, 0, "unknown in the placeholder");
    assert_eq!(planes[0], pcm);
    // ...and patching the final STREAMINFO in at offset 8 gives the full one.
    stream[8..42].copy_from_slice(&info.to_bytes());
    assert_eq!(decode(&stream).unwrap().0.to_bytes(), info.to_bytes());
}

#[test]
fn metadata_block_is_what_containers_carry() {
    let pcm = signal(5000, 2, 16, 4);
    let (_, info) = streamed(48_000, 2, 16, 5, 4096, &pcm, 1);
    let block = info.metadata_block();
    assert_eq!(
        &block[..4],
        &[0x80, 0, 0, 34],
        "last block, STREAMINFO, 34 bytes"
    );
    assert_eq!(&block[4..], &info.to_bytes());
    assert_eq!(&info.stream_header()[..4], b"fLaC");
    assert_eq!(&info.stream_header()[4..], &block);
}

#[test]
fn thirty_two_bit_streams_frame_by_frame() {
    // FLAC 1.4 allows 32 bits; full-scale anti-phase needs a 33-bit side.
    let n = 10_000;
    let pcm: Vec<i32> = (0..n)
        .flat_map(|i| {
            let v = if i % 2 == 0 {
                i32::MAX - i as i32
            } else {
                i32::MIN + i as i32
            };
            [v, v.wrapping_neg().max(i32::MIN + 1)]
        })
        .collect();
    let (frames, info) = streamed(48_000, 2, 32, 5, 4096, &pcm, 6);
    assert_eq!(info.bits_per_sample, 32);
    let mut stream = info.stream_header();
    frames
        .iter()
        .for_each(|f| stream.extend_from_slice(&f.data));
    let (_, planes) = decode(&stream).unwrap();
    let back: Vec<i32> = (0..n).flat_map(|i| [planes[0][i], planes[1][i]]).collect();
    assert_eq!(back, pcm);
}

#[test]
fn the_block_size_is_fixed_once_frames_are_out() {
    let pcm = signal(10_000, 1, 16, 2);
    let mut e = Encoder::new(48_000, 1, 16).unwrap();
    e.push_interleaved(&pcm[..5000]).unwrap();
    assert_eq!(e.take_frames().len(), 1);
    e.set_block_size(1024).unwrap(); // too late for this stream
    e.push_interleaved(&pcm[5000..]).unwrap();
    let rest = e.take_frames();
    assert!(rest.iter().all(|f| f.samples == 4096));
    let (_, info) = e.finish_frames();
    assert_eq!((info.min_block_size, info.max_block_size), (4096, 4096));
    assert_eq!(info.total_samples, 10_000);
}

#[test]
fn finish_after_take_frames_writes_the_header_and_the_rest() {
    let pcm = signal(12_000, 2, 16, 12);
    let want = whole(48_000, 2, 16, 5, 4096, &pcm);
    let mut e = Encoder::new(48_000, 2, 16).unwrap();
    e.set_compression_level(5);
    e.push_interleaved(&pcm).unwrap();
    let taken = e.take_frames();
    let tail = e.finish();
    // finish = final header + frames not yet taken; the taken ones go between.
    let header_len = 42;
    let mut stream = tail[..header_len].to_vec();
    taken.iter().for_each(|f| stream.extend_from_slice(&f.data));
    stream.extend_from_slice(&tail[header_len..]);
    assert_eq!(stream, want);
}
