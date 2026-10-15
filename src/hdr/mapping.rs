//! [`SdrMapping`] — the one analysis mapping the [module note](super) states,
//! and the vocabulary around it.

use core::fmt;

use mediaframe::color::ChromaCoord;
use thiserror::Error;

use super::{
  HLG_NOMINAL_PEAK_NITS, KNEE_START, PQ_PEAK_NITS, REFERENCE_WHITE_NITS, SDR_DISPLAY_GAMMA, expf32,
  hlg_inverse_oetf, hlg_system_gamma, powf32, pq_eotf,
};
use crate::{ColorSpec, Primaries, Transfer};

/// The two BT.2100 transfers [`SdrMapping`] maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HdrTransfer {
  /// SMPTE ST 2084, the Perceptual Quantizer: absolute display light.
  Pq,
  /// ARIB STD-B67, Hybrid Log-Gamma: relative scene light.
  Hlg,
}

impl HdrTransfer {
  /// The HDR transfer `transfer` names, or [`None`] for every other transfer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn of(transfer: &Transfer) -> Option<Self> {
    match transfer {
      Transfer::SmpteSt2084Pq => Some(Self::Pq),
      Transfer::AribStdB67Hlg => Some(Self::Hlg),
      _ => None,
    }
  }

  /// The FFmpeg slug [`Transfer`] spells this transfer with: `"smpte2084"` or
  /// `"arib-std-b67"`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Pq => "smpte2084",
      Self::Hlg => "arib-std-b67",
    }
  }
}

impl fmt::Display for HdrTransfer {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(self.as_str())
  }
}

/// An HDR picture whose gamut [`SdrMapping`] has no BT.709 conversion for.
///
/// The conversion is derived from the gamut's chromaticities and needs it to
/// share BT.709's D65 white. A gamut with another white (BT.470 System M,
/// generic film, DCI-P3, SMPTE ST 428-1) needs a chromatic adaptation this
/// tier does not perform. A gamut with no defined primaries (an unnamed
/// `Other`) has nothing to derive from. The refusal names both the transfer
/// and the gamut.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Error)]
#[error(
  "pixon has no BT.709 conversion for a {transfer} picture over {primaries} primaries: \
   the HDR to SDR mapping converts D65-white RGB gamuts only"
)]
pub struct UnsupportedPrimaries {
  transfer: HdrTransfer,
  primaries: Primaries,
}

impl UnsupportedPrimaries {
  /// The transfer the refused picture declares.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn transfer(&self) -> HdrTransfer {
    self.transfer
  }

  /// The gamut the refused picture declares.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn primaries(&self) -> &Primaries {
    &self.primaries
  }
}

/// Why [`SdrMapping::map_rgb16`] refused its buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
#[non_exhaustive]
pub enum MapError {
  /// The two buffers do not hold the same whole number of `R'G'B'` pixels.
  #[error(
    "the u16 R'G'B' holds {samples} samples and the u8 output {bytes} bytes; \
     both must hold the same whole number of three-channel pixels"
  )]
  LengthMismatch {
    /// Samples in the `u16` input.
    samples: usize,
    /// Bytes in the `u8` output.
    bytes: usize,
  },
  /// The sample depth is outside `1..=16`.
  #[error("{bits} is not a sample depth a u16 R'G'B' buffer holds (1 to 16)")]
  Depth {
    /// The depth that was asked for.
    bits: u32,
  },
}

/// **THE ANALYSIS MAPPING**: an HDR picture's `R'G'B'` to BT.709 SDR `R'G'B'`.
///
/// Built once per picture description, then applied to its pixels. The
/// [module note](super) states the five steps and their numbers. Every
/// transfer other than PQ and HLG builds a passthrough, so a caller can build
/// a mapping for every picture and skip the work where
/// [`is_passthrough`](Self::is_passthrough) says there is none.
///
/// ```
/// use pixon::{Primaries, Transfer, hdr::SdrMapping};
///
/// // An 8-bit HLG pixel at 75%, BT.2408's HDR reference white, over BT.2020.
/// let mapping = SdrMapping::for_picture(&Transfer::AribStdB67Hlg, &Primaries::Bt2020)?;
/// let mut rgb = [191u8, 191, 191];
/// mapping.map_rgb8(&mut rgb);
/// assert_eq!(rgb, [245, 245, 245]); // 0.961 of SDR full scale
///
/// // An SDR picture is left as it is.
/// let sdr = SdrMapping::for_picture(&Transfer::Bt709, &Primaries::Bt709)?;
/// assert!(sdr.is_passthrough());
/// # Ok::<(), pixon::hdr::UnsupportedPrimaries>(())
/// ```
#[derive(Clone, PartialEq)]
pub struct SdrMapping {
  pub(super) hdr: Option<Hdr>,
}

/// The state of a mapping that maps.
#[derive(Clone, PartialEq)]
pub(super) struct Hdr {
  transfer: HdrTransfer,
  /// Linear light in the declared gamut to linear BT.709, row-major.
  pub(super) to_bt709: [[f32; 3]; 3],
  /// The declared gamut's luminance row, the `Y` of its RGB → XYZ matrix,
  /// for the HLG OOTF.
  pub(super) luminance: [f32; 3],
  /// The HLG OOTF's exponent on scene luminance, `γ − 1` at
  /// [`HLG_NOMINAL_PEAK_NITS`].
  ootf_exponent: f32,
  /// The 8-bit face's per-channel linearisation, indexed by code. PQ holds
  /// light relative to reference white; HLG holds scene light, because its
  /// OOTF mixes the channels and runs per pixel.
  table: [f32; 256],
}

impl SdrMapping {
  /// The mapping for a picture declaring `transfer` over `primaries`.
  ///
  /// A PQ or HLG transfer maps. Every other transfer builds a passthrough,
  /// whatever the gamut.
  ///
  /// # Errors
  ///
  /// [`UnsupportedPrimaries`] for a PQ or HLG picture over a gamut with no
  /// derivable BT.709 conversion. See that type for which gamuts those are.
  /// [`Primaries::Unspecified`] is read as BT.2020, the gamut BT.2100 defines
  /// both transfers over.
  pub fn for_picture(
    transfer: &Transfer,
    primaries: &Primaries,
  ) -> Result<Self, UnsupportedPrimaries> {
    let Some(hdr) = HdrTransfer::of(transfer) else {
      return Ok(Self::passthrough());
    };
    let gamut = match primaries {
      Primaries::Unspecified => Primaries::Bt2020,
      other => other.clone(),
    };
    let refused = || UnsupportedPrimaries {
      transfer: hdr,
      primaries: primaries.clone(),
    };
    let to_xyz = rgb_to_xyz(&gamut).ok_or_else(refused)?;
    let bt709_to_xyz = rgb_to_xyz(&Primaries::Bt709).ok_or_else(refused)?;
    let xyz_to_bt709 = invert(&bt709_to_xyz).ok_or_else(refused)?;
    let to_bt709 = narrow(&multiply(&xyz_to_bt709, &to_xyz));
    let luminance = [
      to_xyz[1][0] as f32,
      to_xyz[1][1] as f32,
      to_xyz[1][2] as f32,
    ];
    let table = core::array::from_fn(|code| {
      let signal = code as f32 / 255.0;
      match hdr {
        HdrTransfer::Pq => pq_light(signal),
        HdrTransfer::Hlg => hlg_inverse_oetf(signal),
      }
    });
    Ok(Self {
      hdr: Some(Hdr {
        transfer: hdr,
        to_bt709,
        luminance,
        ootf_exponent: hlg_system_gamma(HLG_NOMINAL_PEAK_NITS) - 1.0,
        table,
      }),
    })
  }

  /// [`for_picture`](Self::for_picture) over the transfer and gamut `spec`
  /// carries.
  ///
  /// # Errors
  ///
  /// [`UnsupportedPrimaries`], as [`for_picture`](Self::for_picture).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn from_spec(spec: &ColorSpec) -> Result<Self, UnsupportedPrimaries> {
    Self::for_picture(&spec.transfer(), &spec.primaries())
  }

  /// The mapping that maps nothing: every face leaves its pixels as they are.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn passthrough() -> Self {
    Self { hdr: None }
  }

  /// Whether this mapping leaves pixels as they are, because the picture
  /// declares no HDR transfer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_passthrough(&self) -> bool {
    self.hdr.is_none()
  }

  /// The HDR transfer this mapping converts from, or [`None`] for a
  /// passthrough.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn transfer(&self) -> Option<HdrTransfer> {
    self.hdr.as_ref().map(|hdr| hdr.transfer)
  }

  /// Maps one `R'G'B'` triple, each channel a signal in `[0, 1]`, to BT.709
  /// SDR `R'G'B'` in `[0, 1]`.
  ///
  /// A passthrough answers its input unchanged. A mapping clamps each channel
  /// to `[0, 1]` first, reading a NaN as `0`.
  #[must_use]
  pub fn map_signal(&self, rgb: [f32; 3]) -> [f32; 3] {
    let Some(hdr) = &self.hdr else {
      return rgb;
    };
    let rgb = rgb.map(unit);
    let light = match hdr.transfer {
      HdrTransfer::Pq => rgb.map(pq_light),
      HdrTransfer::Hlg => hdr.hlg_light(rgb.map(hlg_inverse_oetf)),
    };
    hdr.display(light)
  }

  /// Maps packed 8-bit `R'G'B'` in place, the layout the
  /// [`Convert`](crate::Convert) tier's [`rgb`](crate::Convert::rgb) output
  /// writes. Each byte is a full-range signal, `code / 255`.
  ///
  /// Each complete three-byte pixel is mapped; a trailing partial pixel is
  /// left as it is. A passthrough changes nothing.
  pub fn map_rgb8(&self, rgb: &mut [u8]) {
    let Some(hdr) = &self.hdr else {
      return;
    };
    for pixel in rgb.chunks_exact_mut(3) {
      let linear = [
        hdr.table[pixel[0] as usize],
        hdr.table[pixel[1] as usize],
        hdr.table[pixel[2] as usize],
      ];
      let light = match hdr.transfer {
        HdrTransfer::Pq => linear,
        HdrTransfer::Hlg => hdr.hlg_light(linear),
      };
      let sdr = hdr.display(light);
      pixel[0] = quantize(sdr[0]);
      pixel[1] = quantize(sdr[1]);
      pixel[2] = quantize(sdr[2]);
    }
  }

  /// Maps packed `R'G'B'` at a native depth of `bits` (the layout
  /// `MixedSinker::with_rgb_u16` writes, `0` to `2^bits − 1`) into packed
  /// 8-bit SDR `R'G'B'` in `out`.
  ///
  /// A passthrough narrows each sample to 8 bits (`round(code · 255 /
  /// (2^bits − 1))`) and changes nothing else. A sample above
  /// `2^bits − 1` reads as full scale.
  ///
  /// # Errors
  ///
  /// [`MapError::Depth`] when `bits` is outside `1..=16`, and
  /// [`MapError::LengthMismatch`] when `rgb` and `out` differ in length or
  /// do not hold whole three-channel pixels. Nothing is written on an error.
  pub fn map_rgb16(&self, rgb: &[u16], bits: u32, out: &mut [u8]) -> Result<(), MapError> {
    if !(1..=16).contains(&bits) {
      return Err(MapError::Depth { bits });
    }
    if rgb.len() != out.len() || !rgb.len().is_multiple_of(3) {
      return Err(MapError::LengthMismatch {
        samples: rgb.len(),
        bytes: out.len(),
      });
    }
    let full = ((1u32 << bits) - 1) as f32;
    let signal = |code: u16| (code as f32 / full).min(1.0);
    for (pixel, sdr) in rgb.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
      let mapped = self.map_signal([signal(pixel[0]), signal(pixel[1]), signal(pixel[2])]);
      sdr[0] = quantize(mapped[0]);
      sdr[1] = quantize(mapped[1]);
      sdr[2] = quantize(mapped[2]);
    }
    Ok(())
  }
}

impl fmt::Debug for SdrMapping {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match &self.hdr {
      None => f.write_str("SdrMapping::Passthrough"),
      Some(hdr) => f
        .debug_struct("SdrMapping")
        .field("transfer", &hdr.transfer)
        .field("to_bt709", &hdr.to_bt709)
        .field("luminance", &hdr.luminance)
        .finish_non_exhaustive(),
    }
  }
}

impl Hdr {
  /// HLG scene light to display light relative to reference white: the OOTF
  /// at [`HLG_NOMINAL_PEAK_NITS`], over the declared gamut's luminance.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn hlg_light(&self, scene: [f32; 3]) -> [f32; 3] {
    let ys =
      self.luminance[0] * scene[0] + self.luminance[1] * scene[1] + self.luminance[2] * scene[2];
    if ys <= 0.0 {
      return [0.0; 3];
    }
    let gain = HLG_NOMINAL_PEAK_NITS * powf32(ys, self.ootf_exponent) / REFERENCE_WHITE_NITS;
    [gain * scene[0], gain * scene[1], gain * scene[2]]
  }

  /// Display light relative to reference white, in the declared gamut, to
  /// SDR `R'G'B'`: the gamut matrix, the clip at zero, the knee and the
  /// inverse BT.1886 encode.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn display(&self, light: [f32; 3]) -> [f32; 3] {
    let m = &self.to_bt709;
    let bt709 = [
      m[0][0] * light[0] + m[0][1] * light[1] + m[0][2] * light[2],
      m[1][0] * light[0] + m[1][1] * light[1] + m[1][2] * light[2],
      m[2][0] * light[0] + m[2][1] * light[1] + m[2][2] * light[2],
    ];
    bt709.map(|channel| encode(knee(channel.max(0.0))))
  }
}

/// PQ signal to display light relative to reference white.
#[cfg_attr(not(tarpaulin), inline(always))]
fn pq_light(signal: f32) -> f32 {
  pq_eotf(signal) * (PQ_PEAK_NITS / REFERENCE_WHITE_NITS)
}

/// The knee: identity up to [`KNEE_START`], then an exponential shoulder
/// with slope one at the start and `1.0` as its asymptote.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn knee(light: f32) -> f32 {
  if light <= KNEE_START {
    light
  } else {
    let room = 1.0 - KNEE_START;
    KNEE_START + room * (1.0 - expf32(-(light - KNEE_START) / room))
  }
}

/// SDR display light to signal: the inverse of the BT.1886 EOTF.
#[cfg_attr(not(tarpaulin), inline(always))]
fn encode(light: f32) -> f32 {
  powf32(light.min(1.0), 1.0 / SDR_DISPLAY_GAMMA)
}

/// A signal clamped to `[0, 1]`, a NaN read as `0`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn unit(signal: f32) -> f32 {
  if signal > 0.0 { signal.min(1.0) } else { 0.0 }
}

/// A `[0, 1]` signal to a full-range 8-bit code, rounded to nearest.
#[cfg_attr(not(tarpaulin), inline(always))]
fn quantize(signal: f32) -> u8 {
  (signal * 255.0 + 0.5) as u8
}

/// The RGB → XYZ matrix of a D65-white RGB gamut, from the chromaticities
/// `mediaframe` tabulates (ITU-T H.273, as FFmpeg's `av_csp_primaries_desc`
/// lists them). [`None`] for a gamut with no defined primaries, a CIE XYZ
/// encoding, or a white other than D65.
///
/// The standard construction: each primary's XYZ at unit `Y`, scaled so the
/// three sum to the white point.
pub(super) fn rgb_to_xyz(gamut: &Primaries) -> Option<[[f64; 3]; 3]> {
  if gamut.is_cie_xyz() || gamut.white_point() != Primaries::Bt709.white_point() {
    return None;
  }
  let [red, green, blue] = gamut.chromaticities()?;
  let white = gamut.white_point()?;
  let unit_y = |c: ChromaCoord| {
    let (x, y) = (f64::from(c.x()) / 50_000.0, f64::from(c.y()) / 50_000.0);
    [x / y, 1.0, (1.0 - x - y) / y]
  };
  let (r, g, b, w) = (unit_y(red), unit_y(green), unit_y(blue), unit_y(white));
  let primaries = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
  let scale = apply(&invert(&primaries)?, w);
  Some(core::array::from_fn(|row| {
    core::array::from_fn(|column| primaries[row][column] * scale[column])
  }))
}

/// A 3×3 inverse by cofactors, or [`None`] for a singular matrix.
fn invert(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
  let cofactor =
    |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
  let c = [
    [
      cofactor(1, 2, 1, 2),
      -cofactor(1, 2, 0, 2),
      cofactor(1, 2, 0, 1),
    ],
    [
      -cofactor(0, 2, 1, 2),
      cofactor(0, 2, 0, 2),
      -cofactor(0, 2, 0, 1),
    ],
    [
      cofactor(0, 1, 1, 2),
      -cofactor(0, 1, 0, 2),
      cofactor(0, 1, 0, 1),
    ],
  ];
  let determinant = m[0][0] * c[0][0] + m[0][1] * c[0][1] + m[0][2] * c[0][2];
  if determinant == 0.0 || !determinant.is_finite() {
    return None;
  }
  Some(core::array::from_fn(|row| {
    core::array::from_fn(|column| c[column][row] / determinant)
  }))
}

/// `a · b` for two 3×3 matrices.
fn multiply(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
  core::array::from_fn(|row| {
    core::array::from_fn(|column| (0..3).map(|k| a[row][k] * b[k][column]).sum())
  })
}

/// `m · v`.
fn apply(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
  core::array::from_fn(|row| m[row][0] * v[0] + m[row][1] * v[1] + m[row][2] * v[2])
}

/// A matrix narrowed to `f32` for the per-pixel stage.
fn narrow(m: &[[f64; 3]; 3]) -> [[f32; 3]; 3] {
  core::array::from_fn(|row| core::array::from_fn(|column| m[row][column] as f32))
}
