use rusty_flac::{decode, Encoder};
use std::io::Write;
use std::process::{Command, Stdio};

/// `flac -t` on a stream: `None` when no `flac` binary is on PATH.
fn flac_t(stream: &[u8]) -> Option<bool> {
    let mut child = Command::new("flac")
        .args(["-t", "-s", "-"])
        .stdin(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take().unwrap().write_all(stream).unwrap();
    Some(child.wait().unwrap().success())
}

/// Odd block sizes (the minimum, the 1152/4608 subset codes sent explicitly,
/// the maximum) at every depth path and a narrow, default and wide level
/// round-trip through our decoder and `flac -t`, with a short last block.
#[test]
fn odd_block_sizes_round_trip() {
    let mut flac_ran = false;
    for bs in [16usize, 1152, 4608, 65535] {
        for bps in [16u32, 24, 32] {
            for level in [0u32, 5, 8] {
                let n = 2 * bs + 7;
                let amp = (1i64 << (bps - 2)) as f64;
                let planes: Vec<Vec<i32>> = (0..2)
                    .map(|c| {
                        (0..n)
                            .map(|i| {
                                let t = i as f64 / 44100.0 + c as f64 * 0.3;
                                let tone = (t * 997.0 * std::f64::consts::TAU).sin();
                                let hiss = ((i * 7919 + c * 104729) % 2003) as f64 - 1001.0;
                                (tone * amp + hiss * 13.0) as i32
                            })
                            .collect()
                    })
                    .collect();
                let mut enc = Encoder::new(44100, 2, bps).unwrap();
                enc.set_compression_level(level);
                enc.set_block_size(bs as u32).unwrap();
                enc.push_planar(&[&planes[0], &planes[1]]).unwrap();
                let stream = enc.finish();
                let ctx = format!("bs={bs} bps={bps} level={level}");
                assert_eq!(decode(&stream).unwrap().1, planes, "{ctx}");
                if let Some(ok) = flac_t(&stream) {
                    flac_ran = true;
                    assert!(ok, "flac -t rejected {ctx}");
                }
            }
        }
    }
    if !flac_ran {
        eprintln!("flac not on PATH: skipped the flac -t checks");
    }
}
