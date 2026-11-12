use rusty_flac::{decode, Encoder};

fn lcg(s: &mut u64) -> i64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 33) as i64
}

fn gen(kind: &str, n: usize, ch: usize) -> Vec<Vec<i32>> {
    let mut seed = 7u64;
    (0..ch)
        .map(|c| {
            (0..n)
                .map(|i| {
                    let t = i as f64 / 44100.0;
                    let ph = c as f64 * 0.3;
                    match kind {
                        "sine_noise" => {
                            let v = ((t * 997.0 * std::f64::consts::TAU + ph).sin()
                                * 0.5
                                * 2147483647.0) as i64;
                            (v + (lcg(&mut seed) & 0xFFFF) - 0x8000) as i32
                        }
                        "loud_square" => {
                            if ((i + 17 * c) / 50) % 2 == 0 {
                                i32::MAX
                            } else {
                                i32::MIN
                            }
                        }
                        "extremes" => match i % 4 {
                            0 => i32::MAX,
                            1 => i32::MIN,
                            2 => 0,
                            _ => -1,
                        },
                        "silence" => 0,
                        "const" => -123456789,
                        "wasted24" => {
                            (((t * 440.0 * std::f64::consts::TAU + ph).sin() * 8_000_000.0) as i32)
                                << 8
                        }
                        // clean near-full-scale sine: LPC territory, and close
                        // enough to 2^31 that some predictors overshoot the
                        // 30-bit residual limit and must be skipped
                        "full_sine" => {
                            ((t * 1234.5 * std::f64::consts::TAU + ph).sin() * 2147483000.0) as i32
                        }
                        // smooth, quiet, odd LSBs: LPC on 25-bit-ish values
                        "smooth" => {
                            (((t * 211.0 * std::f64::consts::TAU + ph).sin()
                                + 0.3 * (t * 3301.0 * std::f64::consts::TAU).sin())
                                * 1.0e7) as i32
                                | 1
                        }
                        // 20-bit noise, odd LSBs (no wasted bits): FIXED order 0
                        "quiet_noise" => (lcg(&mut seed) & 0xF_FFFF) as i32 - 0x8_0000,
                        "white" => {
                            (lcg(&mut seed) as i32).wrapping_mul(4) ^ (lcg(&mut seed) as i32)
                        }
                        _ => unreachable!(),
                    }
                })
                .collect()
        })
        .collect()
}

/// Uncorrelated noise with no wasted bits has nothing for a polynomial to
/// predict: FIXED order 0 (Rice-coded samples) must win, at ~20 of 32 bits.
#[test]
fn quiet_noise_takes_fixed_order_0() {
    let planes = gen("quiet_noise", 8192, 1);
    let mut enc = Encoder::new(48000, 1, 32).unwrap();
    enc.push_planar(&[&planes[0]]).unwrap();
    let (flac, stats) = enc.finish_with_stats();
    assert_eq!(stats.fixed_orders[0], stats.frames, "{stats:?}");
    assert!(flac.len() < 8192 * 4 * 21 / 32, "{} bytes", flac.len());
    assert_eq!(decode(&flac).unwrap().1, planes);
}

#[test]
fn roundtrip_and_files() {
    let dir = std::env::var("WIDE_DIR").ok();
    let kinds = [
        "sine_noise",
        "loud_square",
        "extremes",
        "silence",
        "const",
        "wasted24",
        "full_sine",
        "smooth",
        "quiet_noise",
        "white",
    ];
    for kind in kinds {
        for (rate, n, ch) in [
            (44100u32, 20000usize, 2usize),
            (192000, 9000, 2),
            (384000, 4097, 2),
            (44100, 1, 2),
            (44100, 5, 2),
            (96000, 8192, 1),
            (48000, 5000, 3),
            (44100, 4096 + 17, 2),
            (44100, 4096 + 32, 1),
            // A tail of exactly twice the L8 order cap, and the smallest
            // block that gets the segment windows.
            (44100, 4096 + 24, 2),
            (384000, 4096 + 256, 2),
        ] {
            let planes = gen(kind, n, ch);
            let refs: Vec<&[i32]> = planes.iter().map(|v| v.as_slice()).collect();
            for level in [0u32, 5, 8] {
                let mut enc = Encoder::new(rate, ch as u32, 32).unwrap();
                enc.set_compression_level(level);
                enc.push_planar(&refs).unwrap();
                let flac = enc.finish();
                let (info, out) = decode(&flac).unwrap();
                assert_eq!(info.sample_rate, rate);
                assert_eq!(out, planes, "{kind} {rate} {n} {ch} L{level}");
                if let Some(d) = &dir {
                    let name = format!("{d}/{kind}_{rate}_{n}_{ch}_L{level}");
                    std::fs::write(format!("{name}.flac"), &flac).unwrap();
                    let mut raw = Vec::new();
                    for i in 0..n {
                        for plane in &planes {
                            raw.extend_from_slice(&plane[i].to_le_bytes());
                        }
                    }
                    std::fs::write(format!("{name}.s32"), raw).unwrap();
                }
                if n == 20000 && ch == 2 && rate == 44100 {
                    eprintln!("{kind} 20000x2 L{level}: {} bytes of {}", flac.len(), n * 8);
                }
            }
        }
    }
}
