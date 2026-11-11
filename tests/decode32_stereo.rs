//! 32-bit stereo decode against libFLAC: frames whose side channel is 33 bits.

use rusty_flac::decode;

fn lcg(s: &mut u64) -> i64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 33) as i64
}

/// Deterministic 32-bit stereo that pushes libFLAC into every stereo mode.
/// Each 1024-sample frame takes one pattern in turn: quiet clean left with a
/// loud noisy right (left/side), clean right with a noisy left, noise in
/// opposite phase (mid/side), and anti-phase full scale (side up to ~2^32, i.e. a
/// genuinely 33-bit side channel, smooth enough for LPC). Only IEEE add/mul
/// and integer ops, so it regenerates bit-identically on every platform.
fn stereo_modes_signal(n: usize) -> (Vec<i32>, Vec<i32>) {
    let mut seed = 11u64;
    // sin recurrence y[i] = 2c y[i-1] - y[i-2], c = cos(2π·997/44100)
    let c2 = 2.0 * 0.989_928_074_643_058_5_f64;
    let (mut y1, mut y2) = (0.0f64, -0.141_571_208_349_321_6_f64);
    let (mut l, mut r) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let y = c2 * y1 - y2;
        (y2, y1) = (y1, y);
        let mut noise = || (lcg(&mut seed) & 0x3FFF) - 0x2000;
        let (a, b) = match (i / 1024) % 4 {
            0 => {
                // quiet clean left, loud noisy right: left/side is cheapest
                let s = (y * 1.0e6) as i64;
                (s, s + (lcg(&mut seed) & 0xF_FFFF) - 0x8_0000)
            }
            1 => {
                let s = (y * 1.5e9) as i64;
                (s + noise(), s)
            }
            2 => {
                let s = (y * 1.5e9) as i64;
                let d = noise();
                (s + d, s - d)
            }
            _ => {
                let s = (y * 2.1e9) as i64;
                (s, -s - 1)
            }
        };
        l.push(a as i32);
        r.push(b as i32);
    }
    (l, r)
}

/// A libFLAC 1.5.0 stream (`flac -8 -b 1024`) of [`stereo_modes_signal`]:
/// left/side, side/right and mid/side frames, 33-bit side subframes among
/// them. Decoding it is the independent check of the wide side path.
#[test]
fn decodes_libflac_32bit_stereo_modes() {
    let (l, r) = stereo_modes_signal(8192);
    if let Ok(path) = std::env::var("WRITE_STEREO_MODES_S32") {
        let raw: Vec<u8> = l
            .iter()
            .zip(&r)
            .flat_map(|(a, b)| a.to_le_bytes().into_iter().chain(b.to_le_bytes()))
            .collect();
        std::fs::write(path, raw).unwrap();
    }
    let flac = include_bytes!("data/flac150_s32_stereo_modes.flac");
    let (info, out) = decode(flac).unwrap();
    assert_eq!(info.bits_per_sample, 32);
    assert_eq!(out[0], l, "left");
    assert_eq!(out[1], r, "right");
}
