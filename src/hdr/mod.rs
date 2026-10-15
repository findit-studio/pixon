//! **Analysis-grade HDR → SDR**: the ITU-R BT.2100 transfer curves, and one
//! documented mapping from a PQ or HLG picture's `R'G'B'` to the BT.709 SDR
//! `R'G'B'` an analysis consumer reads.
//!
//! # Why this tier exists
//!
//! Every decode pixon performs is transfer-preserving. The affine YCbCr
//! decode is a matrix and a quantisation range, and the non-affine decodes
//! (ICtCp, SMPTE ST 2085, IPT-C2) re-encode through the source's own
//! transfer. So a picture tagged SMPTE ST 2084 (PQ) or ARIB STD-B67 (HLG)
//! decodes to `R'G'B'` that still holds HDR codes. A consumer that reads
//! ordinary display values misreads them: a scene detector's content
//! metric, an image embedding, a keyframe written as a JPEG. ITU-R BT.2408
//! places HDR reference white at PQ 58% and HLG 75%, so such a picture reads
//! dark and flat across its whole diffuse range.
//!
//! [`SdrMapping`] is the conversion that consumer needs. It turns those codes
//! into BT.709 SDR `R'G'B'` through a fixed sequence of steps, and every step
//! is a published curve or matrix, stated below with its numbers.
//!
//! # What it is not
//!
//! It is not render-grade tone mapping. It reads no mastering metadata
//! (`MaxCLL`, `MaxFALL`, SMPTE ST 2086), analyses no scene, and applies one
//! fixed knee rather than a display-adaptive curve such as the BT.2390 EETF.
//! A picture meant for a viewer belongs to a renderer's tone mapper. This
//! tier answers the narrower question an analysis pipeline asks: *what would
//! this picture look like as ordinary SDR?* It answers it the same way for
//! every picture and for every caller.
//!
//! # The curves
//!
//! [`pq_eotf`] and [`pq_inverse_eotf`] (SMPTE ST 2084, BT.2100 Table 4), and
//! [`hlg_oetf`] and [`hlg_inverse_oetf`] (ARIB STD-B67, BT.2100 Table 5),
//! each per channel and in `f32`. [`hlg_system_gamma`] and [`hlg_ootf`] are
//! HLG's display mapping, the step that turns scene light into display light.
//! The constants are BT.2100-2's own, cross-checked against the
//! `colour-science` Python library, and this module's tests pin every one.
//!
//! Each curve clamps its signal domain to the published `[0, 1]`, so a
//! super-white saturates at the peak rather than crossing PQ's pole or
//! growing HLG's log segment without bound. Each is odd-extended through the
//! origin, so a super-black linearises symmetrically.
//!
//! # The mapping, step by step
//!
//! 1. **Linearise to display light.** PQ is absolute: [`pq_eotf`] times
//!    [`PQ_PEAK_NITS`] (10 000 cd/m²). HLG is relative to a display, so the
//!    mapping fixes one: [`hlg_inverse_oetf`] to scene light, then
//!    [`hlg_ootf`] at the nominal peak [`HLG_NOMINAL_PEAK_NITS`] (1000 cd/m²,
//!    BT.2100's reference display, system gamma 1.2). The OOTF's luminance
//!    is the declared gamut's own Y row (BT.2020: `0.2627, 0.6780, 0.0593`).
//! 2. **Normalise to SDR white.** Divide by [`REFERENCE_WHITE_NITS`]
//!    (203 cd/m², BT.2408's HDR reference white), so `1.0` is SDR reference
//!    white. HLG 75% and PQ 58% both land there.
//! 3. **Convert the gamut.** Apply the exact matrix from the declared
//!    primaries to BT.709, derived from the chromaticities `mediaframe`
//!    tabulates (H.273). From BT.2020 it is BT.2087's matrix, and from BT.709
//!    it is the identity. An unspecified gamut reads as BT.2020, because
//!    BT.2100 defines both transfers over BT.2020 primaries. The derivation
//!    needs both gamuts to share D65 white. A gamut with any other white
//!    (BT.470 System M, generic film, DCI-P3, SMPTE ST 428-1's XYZ) or with
//!    no defined primaries is refused by name ([`UnsupportedPrimaries`]),
//!    because it needs a chromatic adaptation this tier does not perform.
//!    A colour outside the BT.709 gamut clips to zero in the channel it
//!    leaves.
//! 4. **Knee**, per channel. Light up to [`KNEE_START`] (0.75 of SDR white)
//!    passes unchanged. Above it, `K + (1 − K)(1 − e^(−(L − K)/(1 − K)))`
//!    compresses every highlight into the last quarter. The curve has slope
//!    one at `K` and approaches `1.0` without clipping. Only `f32` rounding
//!    brings a far highlight to exactly `1.0`. With this knee, HDR reference
//!    white lands at 0.908 of SDR white.
//! 5. **Encode for an SDR display** with the inverse of the BT.1886 EOTF,
//!    `E' = L^(1/2.4)` ([`SDR_DISPLAY_GAMMA`]). Reference white encodes at
//!    0.961, and BT.2408's 18% grey (26 cd/m²) at 0.425.
//!
//! The encode is the BT.1886 inverse and not the BT.709 OETF on purpose.
//! Steps 1 and 2 already yield *display* light, because PQ is display-referred
//! and HLG has passed its OOTF. The BT.709 OETF is a camera curve that
//! expects *scene* light. Applied to display light it would put 18% grey at
//! 0.337 instead of 0.425, while an SDR camera puts it at 0.409. pixon
//! itself reads SDR video `R'G'B'` as BT.1886 display values
//! ([`TransferFunction::for_matrix`](crate::resample::TransferFunction::for_matrix)
//! resolves every YCbCr matrix to BT.1886). So an output of this mapping,
//! linearised by pixon's own SDR curve, gives back exactly the light step 4
//! produced.
//!
//! Every transfer other than PQ and HLG builds a **passthrough** mapping
//! ([`SdrMapping::is_passthrough`]): an SDR picture is already what an
//! analysis consumer reads, and the faces leave it untouched.
//!
//! # Faces and cost
//!
//! [`SdrMapping::map_signal`] maps one `R'G'B'` triple in `f32`.
//! [`SdrMapping::map_rgb8`] maps the packed 8-bit `R'G'B'` the
//! [`Convert`](crate::Convert) tier writes, in place.
//! [`SdrMapping::map_rgb16`] maps the native-depth `u16` `R'G'B'` that
//! `MixedSinker::with_rgb_u16` writes, into 8-bit SDR.
//!
//! The mapping is scalar, as the ICtCp decode is, for the same reason: its
//! transcendentals do not fit the integer lanes the affine kernels use. Per
//! pixel it costs one `powf` for the HLG OOTF and three for the encode, plus
//! an `exp` per channel above the knee. The 8-bit face linearises through a
//! 256-entry table built once per mapping. The intended input is an
//! analysis-sized picture, after the [`Convert`](crate::Convert) tier's
//! fused downscale. Downscaling first averages in the HDR signal domain,
//! which is the domain pixon's default encoded averaging uses for SDR too.

mod mapping;

#[cfg(test)]
mod tests;

pub use mapping::{HdrTransfer, MapError, SdrMapping, UnsupportedPrimaries};

/// Absolute peak of the PQ signal: `1.0` from [`pq_eotf`] is 10 000 cd/m²
/// (SMPTE ST 2084).
pub const PQ_PEAK_NITS: f32 = 10_000.0;

/// HDR reference white, 203 cd/m² (ITU-R BT.2408): the diffuse white
/// [`SdrMapping`] maps to SDR reference white. HLG 75% at the nominal peak
/// and PQ 58% both reproduce it.
pub const REFERENCE_WHITE_NITS: f32 = 203.0;

/// The nominal peak [`SdrMapping`] displays HLG at: BT.2100's 1000 cd/m²
/// reference display, where the system gamma is exactly 1.2.
pub const HLG_NOMINAL_PEAK_NITS: f32 = 1000.0;

/// Where [`SdrMapping`]'s knee starts, as a fraction of SDR reference white
/// in linear light. Below it light passes unchanged, and above it highlights
/// roll off towards `1.0`.
pub const KNEE_START: f32 = 0.75;

/// The SDR display gamma [`SdrMapping`] encodes for: the BT.1886 reference
/// EOTF, `L = E'^2.4`.
pub const SDR_DISPLAY_GAMMA: f32 = 2.4;

/// `f32` `powf` through `std`, or through `libm` on a `no_std + alloc`
/// build.
#[cfg_attr(not(tarpaulin), inline(always))]
fn powf32(x: f32, y: f32) -> f32 {
  #[cfg(feature = "std")]
  {
    f32::powf(x, y)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::powf(x, y)
  }
}

/// `f32` natural logarithm, the companion of [`powf32`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn lnf32(x: f32) -> f32 {
  #[cfg(feature = "std")]
  {
    f32::ln(x)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::logf(x)
  }
}

/// `f32` base-10 logarithm, the companion of [`powf32`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn log10f32(x: f32) -> f32 {
  #[cfg(feature = "std")]
  {
    f32::log10(x)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::log10f(x)
  }
}

/// `f32` exponential, the companion of [`powf32`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn expf32(x: f32) -> f32 {
  #[cfg(feature = "std")]
  {
    f32::exp(x)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::expf(x)
  }
}

/// `f32` square root, the companion of [`powf32`]. `f32::sqrt` is a
/// `std`-only intrinsic, so `no_std` routes through `libm::sqrtf`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn sqrtf32(x: f32) -> f32 {
  #[cfg(feature = "std")]
  {
    f32::sqrt(x)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::sqrtf(x)
  }
}

/// PQ exponent `m1 = 2610 / 16384` (BT.2100 Table 4) = `0.159301758`.
const PQ_M1: f32 = 2610.0 / 16384.0;
/// PQ exponent `m2 = 2523 / 4096 × 128` (BT.2100 Table 4) = `78.84375`.
const PQ_M2: f32 = 2523.0 / 4096.0 * 128.0;
/// PQ coefficient `c1 = 3424 / 4096` (BT.2100 Table 4) = `0.8359375`. It
/// equals `c3 − c2 + 1`, so PQ maps signal `1.0` to light `1.0`.
const PQ_C1: f32 = 3424.0 / 4096.0;
/// PQ coefficient `c2 = 2413 / 4096 × 32` (BT.2100 Table 4) = `18.8515625`.
const PQ_C2: f32 = 2413.0 / 4096.0 * 32.0;
/// PQ coefficient `c3 = 2392 / 4096 × 32` (BT.2100 Table 4) = `18.6875`.
const PQ_C3: f32 = 2392.0 / 4096.0 * 32.0;

/// HLG coefficient `a = 0.17883277` (BT.2100 Table 5, ARIB STD-B67).
const HLG_A: f32 = 0.178_832_77;
/// HLG coefficient `b = 1 − 4a = 0.28466892` (BT.2100 Table 5).
const HLG_B: f32 = 0.284_668_92;
/// HLG coefficient `c = 0.5 − a·ln(4a) = 0.55991073` (BT.2100 Table 5). The
/// literal is the nearest `f32`; the published trailing digit is below `f32`
/// precision.
const HLG_C: f32 = 0.559_910_7;

/// **The PQ EOTF** (SMPTE ST 2084, BT.2100 Table 4): signal `E'` to display
/// light, normalised so `1.0` is 10 000 cd/m² ([`PQ_PEAK_NITS`]).
///
/// `Y = (max(E'^(1/m2) − c1, 0) / (c2 − c3·E'^(1/m2)))^(1/m1)`.
///
/// The signal is defined on `[0, 1]`, so its magnitude is clamped to `1.0`:
/// a super-white saturates at the peak instead of reaching the
/// `c2 − c3·E'^(1/m2) = 0` pole near `|E'| ≈ 1.99`. A negative signal is
/// mirrored through the origin.
///
/// ```
/// // BT.2408's HDR reference white is PQ 58%: 203 cd/m².
/// let light = pixon::hdr::pq_eotf(0.5806) * pixon::hdr::PQ_PEAK_NITS;
/// assert!((light - 203.0).abs() < 1.0);
/// ```
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn pq_eotf(signal: f32) -> f32 {
  let vp = powf32(signal.abs().min(1.0), 1.0 / PQ_M2);
  let num = (vp - PQ_C1).max(0.0);
  let den = PQ_C2 - PQ_C3 * vp;
  signal.signum() * powf32((num / den).max(0.0), 1.0 / PQ_M1)
}

/// **The PQ inverse EOTF** (SMPTE ST 2084, BT.2100 Table 4): display light
/// (`1.0` is 10 000 cd/m²) to signal `E'`.
///
/// `E' = ((c1 + c2·Y^m1) / (1 + c3·Y^m1))^m2`.
///
/// For `Y ≥ 0` the base lies in `(0, c2/c3)`, so the power needs no guard. A
/// negative light is mirrored through the origin. Linear `0` encodes to the
/// small non-zero `c1^m2` (about `7.3e-7`), a property of the curve.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn pq_inverse_eotf(light: f32) -> f32 {
  let yp = powf32(light.abs(), PQ_M1);
  light.signum() * powf32((PQ_C1 + PQ_C2 * yp) / (1.0 + PQ_C3 * yp), PQ_M2)
}

/// **The HLG OETF** (ARIB STD-B67, BT.2100 Table 5): scene light `E` in
/// `[0, 1]` to signal `E'`.
///
/// `E' = sqrt(3·E)` for `|E| ≤ 1/12`, and `a·ln(12·E − b) + c` above it.
/// A negative light is mirrored through the origin.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn hlg_oetf(scene: f32) -> f32 {
  let e = scene.abs();
  let v = if e <= 1.0 / 12.0 {
    sqrtf32(3.0 * e)
  } else {
    HLG_A * lnf32(12.0 * e - HLG_B) + HLG_C
  };
  scene.signum() * v
}

/// **The HLG inverse OETF** (ARIB STD-B67, BT.2100 Table 5): signal `E'` to
/// scene light `E` in `[0, 1]`, per channel.
///
/// `E = E'² / 3` for `|E'| ≤ 1/2`, and `(exp((E' − c)/a) + b) / 12` above it.
///
/// This is scene light, not display light. The HLG EOTF is this curve
/// followed by [`hlg_ootf`], whose system gamma mixes the channels through
/// their luminance. The signal is defined on `[0, 1]`, so its magnitude is
/// clamped to `1.0`, and a negative signal is mirrored through the origin.
///
/// ```
/// // BT.2100's breakpoint: HLG signal 0.5 is scene light 1/12.
/// assert!((pixon::hdr::hlg_inverse_oetf(0.5) - 1.0 / 12.0).abs() < 1e-6);
/// ```
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn hlg_inverse_oetf(signal: f32) -> f32 {
  let a = signal.abs().min(1.0);
  let e = if a <= 0.5 {
    a * a / 3.0
  } else {
    (expf32((a - HLG_C) / HLG_A) + HLG_B) / 12.0
  };
  signal.signum() * e
}

/// **The HLG system gamma** for a display of nominal peak `nominal_peak_nits`
/// (BT.2100 Table 5): `γ = 1.2 + 0.42·log10(L_W / 1000)`.
///
/// It is exactly `1.2` at the 1000 cd/m² reference display. BT.2100 states
/// the extension for peaks from 400 to 2000 cd/m²; outside that range the
/// formula still answers, and the result is the caller's to judge.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn hlg_system_gamma(nominal_peak_nits: f32) -> f32 {
  1.2 + 0.42 * log10f32(nominal_peak_nits / 1000.0)
}

/// **The HLG OOTF** (BT.2100 Table 5): scene light `E` (one value per
/// channel, each in `[0, 1]`) to display light in cd/m², for a display of
/// nominal peak `nominal_peak_nits` and black level zero.
///
/// `F_D = L_W · Y_s^(γ − 1) · E`, with `Y_s` the scene luminance under
/// `luminance` (the gamut's RGB → Y row; BT.2020's is `0.2627, 0.6780,
/// 0.0593`) and `γ` from [`hlg_system_gamma`]. A black pixel (`Y_s = 0`) is
/// black on any display.
///
/// ```
/// use pixon::hdr::{HLG_NOMINAL_PEAK_NITS, hlg_inverse_oetf, hlg_ootf};
///
/// // BT.2408: HLG 75% is HDR reference white, 203 cd/m² on a 1000 cd/m² display.
/// let grey = hlg_inverse_oetf(0.75);
/// let shown = hlg_ootf([grey; 3], [0.2627, 0.6780, 0.0593], HLG_NOMINAL_PEAK_NITS);
/// assert!((shown[1] - 203.0).abs() < 1.0);
/// ```
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn hlg_ootf(scene: [f32; 3], luminance: [f32; 3], nominal_peak_nits: f32) -> [f32; 3] {
  let ys = luminance[0] * scene[0] + luminance[1] * scene[1] + luminance[2] * scene[2];
  if ys <= 0.0 {
    return [0.0; 3];
  }
  let gain = nominal_peak_nits * powf32(ys, hlg_system_gamma(nominal_peak_nits) - 1.0);
  [gain * scene[0], gain * scene[1], gain * scene[2]]
}
