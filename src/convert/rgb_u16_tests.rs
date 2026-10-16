//! The `u16` RGB face of [`Convert`] ([`RgbU16Source`]).
//!
//! - **Parity.** [`Convert::rgb_u16`] writes what the hand-assembled sink's
//!   `with_rgb_u16` writes, byte for byte, at native size, through an area
//!   resize and through a filtered one, for one source of every walker arm and
//!   face family. The walk on the hand-assembled side is the source's own
//!   ([`Source::walk_mixed`]), so what the two sides can differ in is exactly
//!   the face's assembly. Whole-table coverage of the trait and its depths is
//!   `source_coverage`'s.
//! - **Depth.** No sample of a family's face exceeds `2^RGB_U16_DEPTH − 1`, and
//!   a white picture reaches it wherever the face is native or full scale.
//! - **SIMD.** Each kernel family's face is bit-exact against the scalar
//!   reference at every width across the SIMD block sizes and their tails.
//! - **Goldens.** 10- and 12-bit BT.2020 chips, limited and full range, pinned
//!   and held within one code of an `f64` reference decode.
//! - **Banding.** A 10-bit PQ ramp keeps more distinct SDR codes through the
//!   native face and [`SdrMapping::map_rgb16`] than through the 8-bit face and
//!   [`SdrMapping::map_rgb8`] — the reason the face exists at this tier.

use crate::{
  ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, DynamicRange, PixelFormat, Primaries,
  Transfer,
  convert::{Convert, FromSpec, RgbU16Source, Source},
  hdr::SdrMapping,
  resample::{AreaResampler, FilteredResampler, Lanczos3, Resampler},
  sinker::{MixedSinker, MixedSinkerError},
};

// ---- fixtures ---------------------------------------------------------------

/// A deterministic sample stream, so every run sees the same frames.
struct Samples(u64);

impl Samples {
  fn new(seed: u64) -> Self {
    Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
  }

  fn next(&mut self) -> u32 {
    self.0 = self
      .0
      .wrapping_mul(6_364_136_223_846_793_005)
      .wrapping_add(1_442_695_040_888_963_407);
    (self.0 >> 33) as u32
  }

  /// `n` samples in `0..=max`.
  fn u16s(&mut self, n: usize, max: u32) -> Vec<u16> {
    (0..n).map(|_| (self.next() % (max + 1)) as u16).collect()
  }

  /// `n` bytes.
  fn u8s(&mut self, n: usize) -> Vec<u8> {
    (0..n).map(|_| self.next() as u8).collect()
  }

  /// `n` words.
  fn u32s(&mut self, n: usize) -> Vec<u32> {
    (0..n)
      .map(|_| (self.next() << 16) ^ (self.next() & 0xFFFF))
      .collect()
  }

  /// `n` floats in `[0, 1.25]`, so the face's clamp is reached.
  fn f32s(&mut self, n: usize) -> Vec<f32> {
    (0..n)
      .map(|_| (self.next() % 1251) as f32 / 1000.0)
      .collect()
  }
}

/// The spec every law here decodes under: BT.2020 non-constant-luminance, at
/// `range`.
fn spec_2020(range: DynamicRange) -> ColorSpec {
  ColorSpec::from_info(
    PixelFormat::Yuv420p10Le,
    ColorInfo::new(
      Primaries::Bt2020,
      Transfer::Unspecified,
      ColorMatrix::Bt2020Ncl,
      range,
      ChromaLocation::Left,
    ),
  )
}

/// The hand-assembled side's walk, as [`Convert::run`] does it: the spec on the
/// sink, then the source's own walk under the options the spec derives.
fn walk<Fr: Source, R: Resampler>(
  frame: &Fr,
  spec: &ColorSpec,
  sink: &mut MixedSinker<'_, Fr::Marker, R>,
) -> Result<(), MixedSinkerError> {
  sink.set_color_spec(spec)?;
  frame.walk_mixed(&Fr::Options::from_spec(spec)?, sink)
}

/// Asserts `Convert`'s `rgb_u16` equals the hand-assembled sink's
/// `with_rgb_u16` — the result and every sample — at native size, through an
/// area resize to `(ow, oh)` and through a Lanczos-3 resize to the same size.
///
/// A macro rather than a function: `with_rgb_u16` is each source's own sink
/// method, not a trait's, so each call site names it on its concrete sink.
macro_rules! assert_rgb_u16_parity {
  ($label:literal, $frame_ty:ty, $frame:expr, ($w:expr, $h:expr) -> ($ow:expr, $oh:expr)) => {{
    let frame: $frame_ty = $frame;
    let spec = spec_2020(DynamicRange::Limited);
    let (w, h, ow, oh): (usize, usize, usize, usize) = ($w, $h, $ow, $oh);

    let mut tier0 = vec![0u16; w * h * 3];
    let ran0 = Convert::from(&frame)
      .spec(spec.clone())
      .rgb_u16(&mut tier0)
      .run();
    let mut tier1 = vec![0u16; w * h * 3];
    let ran1 = MixedSinker::<<$frame_ty as Source>::Marker>::new(w, h)
      .with_rgb_u16(&mut tier1)
      .and_then(|mut sink| walk(&frame, &spec, &mut sink));
    assert_eq!(ran0, ran1, "{} native: result", $label);
    assert_eq!(tier0, tier1, "{} native: samples", $label);
    assert!(
      tier1.iter().any(|&s| s != 0),
      "{} native: a blank picture",
      $label
    );

    let mut tier0 = vec![0u16; ow * oh * 3];
    let ran0 = Convert::from(&frame)
      .spec(spec.clone())
      .resize(ow, oh)
      .rgb_u16(&mut tier0)
      .run();
    let mut tier1 = vec![0u16; ow * oh * 3];
    let ran1 = MixedSinker::<<$frame_ty as Source>::Marker, AreaResampler>::with_resampler(
      w,
      h,
      AreaResampler::to(ow, oh),
    )
    .and_then(|sink| sink.with_rgb_u16(&mut tier1))
    .and_then(|mut sink| walk(&frame, &spec, &mut sink));
    assert_eq!(ran0, ran1, "{} area: result", $label);
    assert_eq!(tier0, tier1, "{} area: samples", $label);

    let mut tier0 = vec![0u16; ow * oh * 3];
    let ran0 = Convert::from(&frame)
      .spec(spec.clone())
      .resize_with(Lanczos3, ow, oh)
      .rgb_u16(&mut tier0)
      .run();
    let mut tier1 = vec![0u16; ow * oh * 3];
    let ran1 =
      MixedSinker::<<$frame_ty as Source>::Marker, FilteredResampler<Lanczos3>>::with_resampler(
        w,
        h,
        FilteredResampler::new(ow, oh, Lanczos3),
      )
      .and_then(|sink| sink.with_rgb_u16(&mut tier1))
      .and_then(|mut sink| walk(&frame, &spec, &mut sink));
    assert_eq!(ran0, ran1, "{} filtered: result", $label);
    assert_eq!(tier0, tier1, "{} filtered: samples", $label);
  }};
}

/// Asserts no sample of `frame`'s face exceeds `2^RGB_U16_DEPTH − 1`, and,
/// where `reaches` is set, that every sample of this white picture equals it.
fn assert_white_at_depth<Fr: RgbU16Source>(label: &str, frame: &Fr, reaches: bool) {
  let (w, h) = frame.dimensions();
  let max = (1u32 << Fr::RGB_U16_DEPTH) - 1;
  let mut rgb = vec![0u16; w * h * 3];
  Convert::from(frame)
    .spec(spec_2020(DynamicRange::Full))
    .rgb_u16(&mut rgb)
    .run()
    .unwrap();
  assert!(
    rgb.iter().all(|&s| u32::from(s) <= max),
    "{label}: a sample above {max}: {rgb:?}"
  );
  if reaches {
    assert!(
      rgb.iter().all(|&s| u32::from(s) == max),
      "{label}: white is not {max}: {rgb:?}"
    );
  }
}

/// Asserts the face SIMD writes equals the scalar reference's, bit for bit.
fn assert_simd_is_scalar<Fr: RgbU16Source>(label: &str, frame: &Fr) {
  let (w, h) = frame.dimensions();
  let spec = spec_2020(DynamicRange::Limited);
  let mut simd = vec![0u16; w * h * 3];
  Convert::from(frame)
    .spec(spec.clone())
    .rgb_u16(&mut simd)
    .run()
    .unwrap();
  let mut scalar = vec![0u16; w * h * 3];
  Convert::from(frame)
    .spec(spec)
    .simd(false)
    .rgb_u16(&mut scalar)
    .run()
    .unwrap();
  assert_eq!(simd, scalar, "{label} at width {w}");
}

/// Every width the SIMD laws sweep: past AVX-512's widest `u16` block twice,
/// so each backend's block, its tail and a second block are all reached.
const WIDTHS: core::ops::RangeInclusive<usize> = 1..=72;

// ---- parity: one source of every walker arm and face family ------------------

#[cfg(feature = "yuv-planar")]
#[test]
fn parity_yuv420p10() {
  let mut s = Samples::new(1);
  let (y, u, v) = (
    s.u16s(8 * 4, 1023),
    s.u16s(4 * 2, 1023),
    s.u16s(4 * 2, 1023),
  );
  assert_rgb_u16_parity!(
    "yuv420p10",
    crate::frame::Yuv420p10Frame<'_>,
    crate::frame::Yuv420p10Frame::new(&y, &u, &v, 8, 4, 8, 4, 4),
    (8, 4) -> (4, 2)
  );
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn parity_p010() {
  let mut s = Samples::new(2);
  let y: Vec<u16> = s.u16s(8 * 4, 1023).into_iter().map(|c| c << 6).collect();
  let uv: Vec<u16> = s.u16s(8 * 2, 1023).into_iter().map(|c| c << 6).collect();
  assert_rgb_u16_parity!(
    "p010",
    crate::frame::P010Frame<'_>,
    crate::frame::P010Frame::new(&y, &uv, 8, 4, 8, 8),
    (8, 4) -> (4, 2)
  );
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn parity_nv20() {
  let mut s = Samples::new(3);
  let (y, uv) = (s.u16s(8 * 4, 1023), s.u16s(8 * 4, 1023));
  assert_rgb_u16_parity!(
    "nv20",
    crate::frame::Nv20Frame<'_, false>,
    crate::frame::Nv20Frame::<false>::new(&y, &uv, 8, 4, 8, 8),
    (8, 4) -> (4, 2)
  );
}

#[cfg(feature = "y2xx")]
#[test]
fn parity_y210() {
  let mut s = Samples::new(4);
  let packed: Vec<u16> = s
    .u16s(8 * 2 * 4, 1023)
    .into_iter()
    .map(|c| c << 6)
    .collect();
  assert_rgb_u16_parity!(
    "y210",
    crate::frame::Y210Frame<'_>,
    crate::frame::Y210Frame::new(&packed, 8, 4, 16),
    (8, 4) -> (4, 2)
  );
}

#[cfg(feature = "v210")]
#[test]
fn parity_v210() {
  let mut s = Samples::new(5);
  let bytes = s.u8s(16 * 2 * 4);
  assert_rgb_u16_parity!(
    "v210",
    crate::frame::V210Frame<'_, false>,
    crate::frame::V210Frame::<false>::new(&bytes, 12, 4, 32),
    (12, 4) -> (6, 2)
  );
}

#[cfg(feature = "yuv-444-packed")]
#[test]
fn parity_xv36() {
  let mut s = Samples::new(6);
  let packed: Vec<u16> = s
    .u16s(6 * 4 * 4, 4095)
    .into_iter()
    .map(|c| c << 4)
    .collect();
  assert_rgb_u16_parity!(
    "xv36",
    crate::frame::Xv36Frame<'_, false>,
    crate::frame::Xv36Frame::<false>::new(&packed, 6, 4, 24),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "yuv-444-packed")]
#[test]
fn parity_v30x() {
  let mut s = Samples::new(7);
  let packed = s.u32s(6 * 4);
  assert_rgb_u16_parity!(
    "v30x",
    crate::frame::V30XFrame<'_>,
    crate::frame::V30XFrame::new(&packed, 6, 4, 6),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "yuva")]
#[test]
fn parity_yuva444p10() {
  let mut s = Samples::new(8);
  let planes: Vec<Vec<u16>> = (0..4).map(|_| s.u16s(6 * 4, 1023)).collect();
  assert_rgb_u16_parity!(
    "yuva444p10",
    crate::frame::Yuva444p10Frame<'_>,
    crate::frame::Yuva444p10Frame::try_new(
      &planes[0], &planes[1], &planes[2], &planes[3], 6, 4, 6, 6, 6, 6
    )
    .unwrap(),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "rgb")]
#[test]
fn parity_rgb48() {
  let mut s = Samples::new(9);
  let px = s.u16s(6 * 3 * 4, 65535);
  assert_rgb_u16_parity!(
    "rgb48",
    crate::frame::Rgb48Frame<'_, false>,
    crate::frame::Rgb48Frame::<false>::new(&px, 6, 4, 18),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "rgb")]
#[test]
fn parity_x2rgb10() {
  let mut s = Samples::new(10);
  let bytes = s.u8s(6 * 4 * 4);
  assert_rgb_u16_parity!(
    "x2rgb10",
    crate::frame::X2Rgb10Frame<'_, false>,
    crate::frame::X2Rgb10Frame::<false>::new(&bytes, 6, 4, 24),
    (6, 4) -> (3, 2)
  );
}

#[cfg(all(feature = "rgb-float", any(feature = "rgb", feature = "yuv-planar")))]
#[test]
fn parity_rgbf32() {
  let mut s = Samples::new(11);
  let px = s.f32s(6 * 3 * 4);
  assert_rgb_u16_parity!(
    "rgbf32",
    crate::frame::Rgbf32Frame<'_, false>,
    crate::frame::Rgbf32Frame::<false>::new(&px, 6, 4, 18),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "gray")]
#[test]
fn parity_gray12() {
  let mut s = Samples::new(12);
  let y = s.u16s(6 * 4, 4095);
  assert_rgb_u16_parity!(
    "gray12",
    crate::frame::Gray12Frame<'_>,
    crate::frame::Gray12Frame::new(&y, 6, 4, 6),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "gray")]
#[test]
fn parity_gray16() {
  let mut s = Samples::new(13);
  let y = s.u16s(6 * 4, 65535);
  assert_rgb_u16_parity!(
    "gray16",
    crate::frame::Gray16Frame<'_, false>,
    crate::frame::Gray16Frame::<false>::new(&y, 6, 4, 6),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "gray")]
#[test]
fn parity_gray32() {
  let mut s = Samples::new(14);
  let y = s.u32s(6 * 4);
  assert_rgb_u16_parity!(
    "gray32",
    crate::frame::Gray32Frame<'_, false>,
    crate::frame::Gray32Frame::<false>::new(&y, 6, 4, 6),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "gray")]
#[test]
fn parity_ya8() {
  let mut s = Samples::new(15);
  let packed = s.u8s(6 * 2 * 4);
  assert_rgb_u16_parity!(
    "ya8",
    crate::frame::Ya8Frame<'_>,
    crate::frame::Ya8Frame::new(&packed, 6, 4, 12),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "gbr")]
#[test]
fn parity_gbrp10() {
  let mut s = Samples::new(16);
  let planes: Vec<Vec<u16>> = (0..3).map(|_| s.u16s(6 * 4, 1023)).collect();
  assert_rgb_u16_parity!(
    "gbrp10",
    crate::frame::Gbrp10Frame<'_>,
    crate::frame::Gbrp10Frame::try_new(&planes[0], &planes[1], &planes[2], 6, 4, 6, 6, 6).unwrap(),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "gbr")]
#[test]
fn parity_gbrpf32() {
  let mut s = Samples::new(17);
  let planes: Vec<Vec<f32>> = (0..3).map(|_| s.f32s(6 * 4)).collect();
  assert_rgb_u16_parity!(
    "gbrpf32",
    crate::frame::Gbrpf32Frame<'_, false>,
    crate::frame::Gbrpf32Frame::<false>::try_new(
      &planes[0], &planes[1], &planes[2], 6, 4, 6, 6, 6
    )
    .unwrap(),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "xyz")]
#[test]
fn parity_xyz12() {
  let mut s = Samples::new(18);
  let px: Vec<u16> = s
    .u16s(6 * 3 * 4, 4095)
    .into_iter()
    .map(|c| c << 4)
    .collect();
  assert_rgb_u16_parity!(
    "xyz12",
    crate::frame::Xyz12Frame<'_, false>,
    crate::frame::Xyz12Frame::<false>::new(&px, 6, 4, 18),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "mono")]
#[test]
fn parity_pal8() {
  let mut s = Samples::new(19);
  let idx = s.u8s(6 * 4);
  let mut palette = [[0u8; 4]; 256];
  for entry in &mut palette {
    let bytes = s.u8s(4);
    entry.copy_from_slice(&bytes);
  }
  assert_rgb_u16_parity!(
    "pal8",
    crate::frame::Pal8Frame<'_>,
    crate::frame::Pal8Frame::new(&idx, &palette, 6, 4, 6),
    (6, 4) -> (3, 2)
  );
}

#[cfg(feature = "mono")]
#[test]
fn parity_monoblack() {
  let mut s = Samples::new(20);
  let bits = s.u8s(2 * 4);
  assert_rgb_u16_parity!(
    "monoblack",
    crate::frame::MonoblackFrame<'_>,
    crate::frame::MonoblackFrame::new(&bits, 12, 4, 2),
    (12, 4) -> (6, 2)
  );
}

// ---- depth: no sample past the face's depth, and white at it ------------------

#[cfg(feature = "yuv-planar")]
#[test]
fn depth_yuv420p10() {
  let (y, uv) = ([1023u16; 8 * 2], [512u16; 4]);
  let frame = crate::frame::Yuv420p10Frame::new(&y, &uv, &uv, 8, 2, 8, 4, 4);
  assert_white_at_depth("yuv420p10", &frame, true);
}

#[cfg(feature = "yuv-planar")]
#[test]
fn depth_yuv444p12() {
  let (y, uv) = ([4095u16; 6 * 2], [2048u16; 6 * 2]);
  let frame = crate::frame::Yuv444p12Frame::new(&y, &uv, &uv, 6, 2, 6, 6, 6);
  assert_white_at_depth("yuv444p12", &frame, true);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn depth_p010() {
  let y = [1023u16 << 6; 8 * 2];
  let uv = [512u16 << 6; 8];
  let frame = crate::frame::P010Frame::new(&y, &uv, 8, 2, 8, 8);
  assert_white_at_depth("p010", &frame, true);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn depth_nv20() {
  let (y, uv) = ([1023u16; 8 * 2], [512u16; 8 * 2]);
  let frame = crate::frame::Nv20Frame::<false>::new(&y, &uv, 8, 2, 8, 8);
  assert_white_at_depth("nv20", &frame, true);
}

#[cfg(feature = "y2xx")]
#[test]
fn depth_y212() {
  // Y U Y V, each 12-bit sample high-packed.
  let packed: Vec<u16> = [4095u16, 2048, 4095, 2048]
    .iter()
    .cycle()
    .take(8 * 2 * 2)
    .map(|&c| c << 4)
    .collect();
  let frame = crate::frame::Y2xxFrame::<12, false>::new(&packed, 8, 2, 16);
  assert_white_at_depth("y212", &frame, true);
}

#[cfg(feature = "yuv-444-packed")]
#[test]
fn depth_xv48() {
  // U Y V A, 16-bit each.
  let packed: Vec<u16> = [32768u16, 65535, 32768, 65535]
    .iter()
    .cycle()
    .take(6 * 4 * 2)
    .copied()
    .collect();
  let frame = crate::frame::Xv48Frame::<false>::new(&packed, 6, 2, 24);
  assert_white_at_depth("xv48", &frame, true);
}

#[cfg(feature = "yuva")]
#[test]
fn depth_yuva420p16() {
  let (y, uv, a) = ([65535u16; 8 * 2], [32768u16; 4], [65535u16; 8 * 2]);
  let frame =
    crate::frame::Yuva420pFrame16::<16, false>::try_new(&y, &uv, &uv, &a, 8, 2, 8, 4, 4, 8)
      .unwrap();
  assert_white_at_depth("yuva420p16", &frame, true);
}

#[cfg(feature = "rgb")]
#[test]
fn depth_x2rgb10() {
  let words = [0x3FFF_FFFFu32; 6 * 2];
  let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
  let frame = crate::frame::X2Rgb10Frame::<false>::new(&bytes, 6, 2, 24);
  assert_white_at_depth("x2rgb10", &frame, true);
}

#[cfg(feature = "rgb")]
#[test]
fn depth_rgba128() {
  let px = [u32::MAX; 6 * 4 * 2];
  let frame = crate::frame::Rgba128Frame::<false>::new(&px, 6, 2, 24);
  assert_white_at_depth("rgba128", &frame, true);
}

#[cfg(all(feature = "rgb-float", any(feature = "rgb", feature = "yuv-planar")))]
#[test]
fn depth_rgbaf32() {
  let px = [1.0f32; 6 * 4 * 2];
  let frame = crate::frame::Rgbaf32Frame::<false>::new(&px, 6, 2, 24);
  assert_white_at_depth("rgbaf32", &frame, true);
}

#[cfg(feature = "gray")]
#[test]
fn depth_gray14() {
  let y = [16383u16; 6 * 2];
  let frame = crate::frame::GrayNFrame::<14, false>::new(&y, 6, 2, 6);
  assert_white_at_depth("gray14", &frame, true);
}

#[cfg(feature = "gray")]
#[test]
fn depth_ya16() {
  let packed = [65535u16; 6 * 2 * 2];
  let frame = crate::frame::Ya16Frame::<false>::new(&packed, 6, 2, 12);
  assert_white_at_depth("ya16", &frame, true);
}

#[cfg(feature = "gray")]
#[test]
fn depth_grayf32() {
  let y = [1.0f32; 6 * 2];
  let frame = crate::frame::Grayf32Frame::<false>::new(&y, 6, 2, 6);
  assert_white_at_depth("grayf32", &frame, true);
}

#[cfg(feature = "gray")]
#[test]
fn depth_ya8() {
  let packed = [255u8; 6 * 2 * 2];
  let frame = crate::frame::Ya8Frame::new(&packed, 6, 2, 12);
  assert_white_at_depth("ya8", &frame, true);
}

#[cfg(feature = "gbr")]
#[test]
fn depth_gbrap16() {
  let plane = [65535u16; 6 * 2];
  let frame = crate::frame::GbrapHighBitFrame::<16, false>::try_new(
    &plane, &plane, &plane, &plane, 6, 2, 6, 6, 6, 6,
  )
  .unwrap();
  assert_white_at_depth("gbrap16", &frame, true);
}

#[cfg(feature = "gbr")]
#[test]
fn depth_gbrp12msb() {
  let plane = [4095u16 << 4; 6 * 2];
  let frame =
    crate::frame::GbrpMsbFrame::<12, false>::try_new(&plane, &plane, &plane, 6, 2, 6, 6, 6)
      .unwrap();
  assert_white_at_depth("gbrp12msb", &frame, true);
}

#[cfg(feature = "gbr")]
#[test]
fn depth_gbrap32() {
  let plane = [u32::MAX; 6 * 2];
  let frame =
    crate::frame::Gbrap32Frame::<false>::try_new(&plane, &plane, &plane, &plane, 6, 2, 6, 6, 6, 6)
      .unwrap();
  assert_white_at_depth("gbrap32", &frame, true);
}

#[cfg(feature = "mono")]
#[test]
fn depth_pal8() {
  let idx = [7u8; 6 * 2];
  let mut palette = [[0u8; 4]; 256];
  palette[7] = [255, 255, 255, 255];
  let frame = crate::frame::Pal8Frame::new(&idx, &palette, 6, 2, 6);
  assert_white_at_depth("pal8", &frame, true);
}

#[cfg(feature = "mono")]
#[test]
fn depth_monowhite() {
  // Monowhite: a clear bit is white.
  let bits = [0u8; 2];
  let frame = crate::frame::MonowhiteFrame::new(&bits, 8, 2, 1);
  assert_white_at_depth("monowhite", &frame, true);
}

/// XYZ12's face is linear RGB at full scale: a sample can stop short of
/// `65535` for a white the gamut does not hold at its corner, so only the
/// ceiling is asserted.
#[cfg(feature = "xyz")]
#[test]
fn depth_xyz12() {
  let px = [4095u16 << 4; 6 * 3 * 2];
  let frame = crate::frame::Xyz12Frame::<false>::new(&px, 6, 2, 18);
  assert_white_at_depth("xyz12", &frame, false);
}

// ---- SIMD against the scalar reference, every width, every kernel family -------

#[cfg(feature = "yuv-planar")]
#[test]
fn simd_is_scalar_yuv420p10() {
  for w in WIDTHS.filter(|w| w % 2 == 0) {
    let mut s = Samples::new(w as u64);
    let (y, u, v) = (
      s.u16s(w * 2, 1023),
      s.u16s(w / 2, 1023),
      s.u16s(w / 2, 1023),
    );
    let frame = crate::frame::Yuv420p10Frame::new(
      &y,
      &u,
      &v,
      w as u32,
      2,
      w as u32,
      (w / 2) as u32,
      (w / 2) as u32,
    );
    assert_simd_is_scalar("yuv420p10", &frame);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn simd_is_scalar_yuv444p12() {
  for w in WIDTHS {
    let mut s = Samples::new(w as u64);
    let (y, u, v) = (
      s.u16s(w * 2, 4095),
      s.u16s(w * 2, 4095),
      s.u16s(w * 2, 4095),
    );
    let frame =
      crate::frame::Yuv444p12Frame::new(&y, &u, &v, w as u32, 2, w as u32, w as u32, w as u32);
    assert_simd_is_scalar("yuv444p12", &frame);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn simd_is_scalar_p010() {
  for w in WIDTHS.filter(|w| w % 2 == 0) {
    let mut s = Samples::new(w as u64);
    let y: Vec<u16> = s.u16s(w * 2, 1023).into_iter().map(|c| c << 6).collect();
    let uv: Vec<u16> = s.u16s(w, 1023).into_iter().map(|c| c << 6).collect();
    let frame = crate::frame::P010Frame::new(&y, &uv, w as u32, 2, w as u32, w as u32);
    assert_simd_is_scalar("p010", &frame);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn simd_is_scalar_p410() {
  for w in WIDTHS {
    let mut s = Samples::new(w as u64);
    let y: Vec<u16> = s.u16s(w * 2, 1023).into_iter().map(|c| c << 6).collect();
    let uv: Vec<u16> = s
      .u16s(w * 2 * 2, 1023)
      .into_iter()
      .map(|c| c << 6)
      .collect();
    let frame = crate::frame::P410Frame::new(&y, &uv, w as u32, 2, w as u32, (w * 2) as u32);
    assert_simd_is_scalar("p410", &frame);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn simd_is_scalar_nv20() {
  for w in WIDTHS.filter(|w| w % 2 == 0) {
    let mut s = Samples::new(w as u64);
    let (y, uv) = (s.u16s(w * 2, 1023), s.u16s(w * 2, 1023));
    let frame = crate::frame::Nv20Frame::<false>::new(&y, &uv, w as u32, 2, w as u32, w as u32);
    assert_simd_is_scalar("nv20", &frame);
  }
}

#[cfg(feature = "y2xx")]
#[test]
fn simd_is_scalar_y210() {
  for w in WIDTHS.filter(|w| w % 2 == 0) {
    let mut s = Samples::new(w as u64);
    let packed: Vec<u16> = s
      .u16s(w * 2 * 2, 1023)
      .into_iter()
      .map(|c| c << 6)
      .collect();
    let frame = crate::frame::Y210Frame::new(&packed, w as u32, 2, (w * 2) as u32);
    assert_simd_is_scalar("y210", &frame);
  }
}

#[cfg(feature = "v210")]
#[test]
fn simd_is_scalar_v210() {
  for w in WIDTHS.filter(|w| w % 2 == 0) {
    let mut s = Samples::new(w as u64);
    let stride = w.div_ceil(6) * 16;
    let bytes = s.u8s(stride * 2);
    let frame = crate::frame::V210Frame::<false>::new(&bytes, w as u32, 2, stride as u32);
    assert_simd_is_scalar("v210", &frame);
  }
}

#[cfg(feature = "yuv-444-packed")]
#[test]
fn simd_is_scalar_xv36() {
  for w in WIDTHS {
    let mut s = Samples::new(w as u64);
    let packed: Vec<u16> = s
      .u16s(w * 4 * 2, 4095)
      .into_iter()
      .map(|c| c << 4)
      .collect();
    let frame = crate::frame::Xv36Frame::<false>::new(&packed, w as u32, 2, (w * 4) as u32);
    assert_simd_is_scalar("xv36", &frame);
  }
}

#[cfg(feature = "yuva")]
#[test]
fn simd_is_scalar_yuva444p10() {
  for w in WIDTHS {
    let mut s = Samples::new(w as u64);
    let planes: Vec<Vec<u16>> = (0..4).map(|_| s.u16s(w * 2, 1023)).collect();
    let n = w as u32;
    let frame = crate::frame::Yuva444p10Frame::try_new(
      &planes[0], &planes[1], &planes[2], &planes[3], n, 2, n, n, n, n,
    )
    .unwrap();
    assert_simd_is_scalar("yuva444p10", &frame);
  }
}

#[cfg(feature = "rgb")]
#[test]
fn simd_is_scalar_rgb48() {
  for w in WIDTHS {
    let mut s = Samples::new(w as u64);
    let px = s.u16s(w * 3 * 2, 65535);
    let frame = crate::frame::Rgb48Frame::<false>::new(&px, w as u32, 2, (w * 3) as u32);
    assert_simd_is_scalar("rgb48", &frame);
  }
}

#[cfg(feature = "rgb")]
#[test]
fn simd_is_scalar_x2rgb10() {
  for w in WIDTHS {
    let mut s = Samples::new(w as u64);
    let bytes = s.u8s(w * 4 * 2);
    let frame = crate::frame::X2Rgb10Frame::<false>::new(&bytes, w as u32, 2, (w * 4) as u32);
    assert_simd_is_scalar("x2rgb10", &frame);
  }
}

#[cfg(all(feature = "rgb-float", any(feature = "rgb", feature = "yuv-planar")))]
#[test]
fn simd_is_scalar_rgbf32() {
  for w in WIDTHS {
    let mut s = Samples::new(w as u64);
    let px = s.f32s(w * 3 * 2);
    let frame = crate::frame::Rgbf32Frame::<false>::new(&px, w as u32, 2, (w * 3) as u32);
    assert_simd_is_scalar("rgbf32", &frame);
  }
}

#[cfg(feature = "gray")]
#[test]
fn simd_is_scalar_gray12() {
  for w in WIDTHS {
    let mut s = Samples::new(w as u64);
    let y = s.u16s(w * 2, 4095);
    let frame = crate::frame::Gray12Frame::<false>::new(&y, w as u32, 2, w as u32);
    assert_simd_is_scalar("gray12", &frame);
  }
}

#[cfg(feature = "gbr")]
#[test]
fn simd_is_scalar_gbrp10() {
  for w in WIDTHS {
    let mut s = Samples::new(w as u64);
    let planes: Vec<Vec<u16>> = (0..3).map(|_| s.u16s(w * 2, 1023)).collect();
    let n = w as u32;
    let frame = crate::frame::Gbrp10Frame::<false>::try_new(
      &planes[0], &planes[1], &planes[2], n, 2, n, n, n,
    )
    .unwrap();
    assert_simd_is_scalar("gbrp10", &frame);
  }
}

#[cfg(feature = "xyz")]
#[test]
fn simd_is_scalar_xyz12() {
  for w in WIDTHS {
    let mut s = Samples::new(w as u64);
    let px: Vec<u16> = s
      .u16s(w * 3 * 2, 4095)
      .into_iter()
      .map(|c| c << 4)
      .collect();
    let frame = crate::frame::Xyz12Frame::<false>::new(&px, w as u32, 2, (w * 3) as u32);
    assert_simd_is_scalar("xyz12", &frame);
  }
}

// ---- goldens: BT.2020 chips, 10 and 12 bit, limited and full ------------------

/// BT.2020 NCL's luma weights.
const KR: f64 = 0.2627;
const KB: f64 = 0.0593;

/// The `R'G'B'` `f64` decode of one 4:4:4 sample at `bits` and `range`, each
/// channel scaled to `0..=2^bits − 1` and rounded — the reference a golden is
/// held to within one code.
fn reference(y: u16, cb: u16, cr: u16, bits: u32, full: bool) -> [u16; 3] {
  let max = f64::from((1u32 << bits) - 1);
  let scale = f64::from(1u32 << (bits - 8));
  let (luma, chroma) = if full {
    (f64::from(y) / max, (f64::from(cb) - 128.0 * scale) / max)
  } else {
    (
      (f64::from(y) - 16.0 * scale) / (219.0 * scale),
      (f64::from(cb) - 128.0 * scale) / (224.0 * scale),
    )
  };
  let red_chroma = if full {
    (f64::from(cr) - 128.0 * scale) / max
  } else {
    (f64::from(cr) - 128.0 * scale) / (224.0 * scale)
  };
  let r = luma + 2.0 * (1.0 - KR) * red_chroma;
  let b = luma + 2.0 * (1.0 - KB) * chroma;
  let g = (luma - KR * r - KB * b) / (1.0 - KR - KB);
  [r, g, b].map(|channel| (channel.clamp(0.0, 1.0) * max).round() as u16)
}

/// The chips: black, white, mid grey and the 75% primaries, as `R'G'B'`
/// fractions, encoded to `Y'CbCr` at `bits` and `range`.
fn chips(bits: u32, full: bool) -> Vec<[u16; 3]> {
  let max = f64::from((1u32 << bits) - 1);
  let scale = f64::from(1u32 << (bits - 8));
  [
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.5, 0.5, 0.5],
    [0.75, 0.0, 0.0],
    [0.0, 0.75, 0.0],
    [0.0, 0.0, 0.75],
  ]
  .iter()
  .map(|&[r, g, b]: &[f64; 3]| {
    let y = KR * r + (1.0 - KR - KB) * g + KB * b;
    let cb = (b - y) / (2.0 * (1.0 - KB));
    let cr = (r - y) / (2.0 * (1.0 - KR));
    let code = |v: f64| v.round() as u16;
    if full {
      [
        code(y * max),
        code(cb * max + 128.0 * scale),
        code(cr * max + 128.0 * scale),
      ]
    } else {
      [
        code(16.0 * scale + 219.0 * scale * y),
        code(128.0 * scale + 224.0 * scale * cb),
        code(128.0 * scale + 224.0 * scale * cr),
      ]
    }
  })
  .collect()
}

/// Decodes the chips through `rgb_u16` and asserts the pinned golden, and that
/// it is within one code of the `f64` reference.
#[cfg(feature = "yuv-planar")]
fn assert_golden(bits: u32, full: bool, golden: &[[u16; 3]; 6]) {
  let chips = chips(bits, full);
  let (y, u, v): (Vec<u16>, Vec<u16>, Vec<u16>) = (
    chips.iter().map(|c| c[0]).collect(),
    chips.iter().map(|c| c[1]).collect(),
    chips.iter().map(|c| c[2]).collect(),
  );
  let range = if full {
    DynamicRange::Full
  } else {
    DynamicRange::Limited
  };
  let mut rgb = vec![0u16; 6 * 3];
  match bits {
    10 => Convert::from(&crate::frame::Yuv444p10Frame::new(
      &y, &u, &v, 6, 1, 6, 6, 6,
    ))
    .spec(spec_2020(range))
    .rgb_u16(&mut rgb)
    .run()
    .unwrap(),
    12 => Convert::from(&crate::frame::Yuv444p12Frame::new(
      &y, &u, &v, 6, 1, 6, 6, 6,
    ))
    .spec(spec_2020(range))
    .rgb_u16(&mut rgb)
    .run()
    .unwrap(),
    _ => unreachable!("the goldens are 10 and 12 bit"),
  }
  for (at, chip) in chips.iter().enumerate() {
    let decoded = [rgb[at * 3], rgb[at * 3 + 1], rgb[at * 3 + 2]];
    let expected = reference(chip[0], chip[1], chip[2], bits, full);
    for channel in 0..3 {
      assert!(
        decoded[channel].abs_diff(expected[channel]) <= 1,
        "{bits}-bit full={full} chip {at}: {decoded:?} against the reference {expected:?}"
      );
    }
    assert_eq!(decoded, golden[at], "{bits}-bit full={full} chip {at}");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn golden_bt2020_10_bit_limited() {
  assert_golden(10, false, &GOLDEN_10_LIMITED);
}

#[cfg(feature = "yuv-planar")]
#[test]
fn golden_bt2020_10_bit_full() {
  assert_golden(10, true, &GOLDEN_10_FULL);
}

#[cfg(feature = "yuv-planar")]
#[test]
fn golden_bt2020_12_bit_limited() {
  assert_golden(12, false, &GOLDEN_12_LIMITED);
}

#[cfg(feature = "yuv-planar")]
#[test]
fn golden_bt2020_12_bit_full() {
  assert_golden(12, true, &GOLDEN_12_FULL);
}

/// Black, white, grey, 75% red, 75% green, 75% blue, as this build decodes
/// them: each channel within one code of the `f64` reference, the code a
/// near-zero chroma rounds to included.
const GOLDEN_10_LIMITED: [[u16; 3]; 6] = [
  [0, 0, 0],
  [1023, 1023, 1023],
  [512, 512, 512],
  [768, 0, 1],
  [0, 767, 1],
  [0, 1, 768],
];
const GOLDEN_10_FULL: [[u16; 3]; 6] = [
  [0, 0, 0],
  [1023, 1023, 1023],
  [512, 512, 512],
  [768, 0, 1],
  [0, 767, 1],
  [0, 0, 767],
];
const GOLDEN_12_LIMITED: [[u16; 3]; 6] = [
  [0, 0, 0],
  [4095, 4095, 4095],
  [2048, 2048, 2048],
  [3071, 0, 1],
  [1, 3072, 0],
  [1, 0, 3072],
];
const GOLDEN_12_FULL: [[u16; 3]; 6] = [
  [0, 0, 0],
  [4095, 4095, 4095],
  [2048, 2048, 2048],
  [3072, 0, 0],
  [0, 3071, 0],
  [0, 0, 3072],
];

// ---- banding: the reason the face exists at this tier -------------------------

/// A 10-bit PQ ramp keeps more distinct SDR codes through the native face than
/// through the 8-bit one.
///
/// The ramp is a neutral grey, `Y'` stepping one code at a time over PQ signal
/// 0.3 to 0.6 (about 6 to 250 cd/m²), limited range over BT.2020. The 8-bit face
/// rounds every three or four of those codes onto one before the mapping sees
/// them; the native face hands each to [`SdrMapping::map_rgb16`] at its own
/// depth. The counts are this build's, pinned.
#[cfg(feature = "yuv-planar")]
#[test]
fn a_10_bit_pq_ramp_keeps_more_sdr_codes_through_the_native_face() {
  let y: Vec<u16> = (327u16..=590).collect();
  let w = y.len();
  let neutral = vec![512u16; w];
  let frame = crate::frame::Yuv444p10Frame::new(
    &y, &neutral, &neutral, w as u32, 1, w as u32, w as u32, w as u32,
  );
  let spec = ColorSpec::from_info(
    PixelFormat::Yuv444p10Le,
    ColorInfo::new(
      Primaries::Bt2020,
      Transfer::SmpteSt2084Pq,
      ColorMatrix::Bt2020Ncl,
      DynamicRange::Limited,
      ChromaLocation::Left,
    ),
  );
  let mapping = SdrMapping::from_spec(&spec).unwrap();

  let mut eight = vec![0u8; w * 3];
  Convert::from(&frame)
    .spec(spec.clone())
    .rgb(&mut eight)
    .run()
    .unwrap();
  mapping.map_rgb8(&mut eight);

  let mut native = vec![0u16; w * 3];
  Convert::from(&frame)
    .spec(spec)
    .rgb_u16(&mut native)
    .run()
    .unwrap();
  let mut mapped = vec![0u8; w * 3];
  mapping
    .map_rgb16(
      &native,
      <crate::frame::Yuv444p10Frame<'_> as RgbU16Source>::RGB_U16_DEPTH,
      &mut mapped,
    )
    .unwrap();

  let distinct = |sdr: &[u8]| {
    let mut codes: Vec<u8> = sdr
      .as_chunks::<3>()
      .0
      .iter()
      .map(|pixel| pixel[1])
      .collect();
    codes.sort_unstable();
    codes.dedup();
    codes.len()
  };
  let (through_eight, through_native) = (distinct(&eight), distinct(&mapped));
  assert!(
    through_native > 2 * through_eight,
    "native {through_native} against eight-bit {through_eight}"
  );
  assert_eq!((through_eight, through_native), BANDING_COUNTS);
}

/// The distinct SDR codes the 264-code ramp keeps: 77 through the 8-bit face,
/// 170 through the native one.
const BANDING_COUNTS: (usize, usize) = (77, 170);
