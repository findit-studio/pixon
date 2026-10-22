//! Tier-0 [`Convert`] — the golden one-call decode entry point.
//!
//! [`Convert`] is the top of pixon's tier stack (issue
//! [#392](https://github.com/findit-studio/pixon/issues/392)): a single
//! generic builder that decodes a validated source frame to one or more
//! output buffers with zero redundant parameters. Dimensions come from the
//! frame, colorimetry comes from a single [`ColorSpec`], and the per-format
//! walk knobs are *derived* from that spec via [`FromSpec`] (overridable).
//! Every setter is infallible; [`Convert::run`] is the only fallible call.
//!
//! ```text
//! Convert::from(&frame).spec(spec).resize(w, h).rgb(&mut rgb).run()?  // Tier 0 — this module
//! MixedSinker::new(w, h).with_rgb(&mut rgb)?; yuv420p_to(&frame, …)?  // Tier 1 — streaming / custom
//! ```
//!
//! Tier 0 is purely additive: it assembles the mid-tier
//! [`MixedSinker`] internally and forwards to the
//! same per-format walkers, so its output is byte-identical to hand-assembling
//! the sink. The mid tier stays the supported home for custom sinks, partial
//! feeding, and bespoke geometry.
//!
//! The [`Source`] contract between frames and [`Convert`] is *sealed*: it is
//! implemented, per source format, by the same `walker!` table that drives the
//! [`Walker`](crate::Walker) tier, so the two can never drift.

#[cfg(all(test, feature = "yuv-planar"))]
mod tests;
// The `u16` RGB face's laws: parity, depth, SIMD, goldens and banding across the
// face families, so they ride the umbrella `frame` feature that enables them all.
#[cfg(all(test, feature = "frame"))]
mod rgb_u16_tests;
// Whole-table `Source` coverage — the compile-time exhaustiveness proof plus
// one byte-parity spot-check per family. Rides the `Source` gate
// (`any(std, alloc)`) and additionally requires at least one convertible source
// feature, so the parity helpers and their imports are never dead on a
// format-free build.
#[cfg(all(
  test,
  any(feature = "std", feature = "alloc"),
  any(
    feature = "xyz",
    feature = "mono",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "yuva",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "gray",
    feature = "gbr",
  )
))]
mod source_coverage;

use crate::{
  SourceFormat,
  resample::{AreaResampler, FilterKernel, FilteredResampler, NoopResampler, Resampler},
  sinker::{MixedSinker, MixedSinkerError},
  walker::{ColorSpec, YuvOptions},
};

// `ChromaLocation` and `St428Interpretation` feed only the `yuv-planar`
// sink-knob setters and their fields, so their imports must ride the same gate
// or they warn `unused_imports` under the other feature powersets (CI builds
// those with `-Dwarnings`).
#[cfg(feature = "yuv-planar")]
use crate::{ChromaLocation, sinker::St428Interpretation};

/// The argument a sink with no `u16` RGB face is probed with, by the
/// `walker!` table's rows that state none: uninhabited, so the probe is never
/// called.
// A feature set whose rows all state a face has no probe to use it.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) enum Faceless {}

/// The `u16` RGB face's census, by construction: a `walker!` row that states
/// no face calls `set_rgb_u16` and `with_rgb_u16` on its sink with a
/// [`Faceless`]. Where the sink has no face, this trait's methods take the
/// call. Where it has one, its own inherent methods take it — an inherent
/// method wins over a trait's — and refuse the argument, so the row does not
/// build until it states the face its sink has.
// As for `Faceless`: a feature set may hold no row without a face.
#[allow(dead_code)]
pub(crate) trait FacelessSink {
  /// Never called: [`Faceless`] has no value.
  fn set_rgb_u16(&mut self, faceless: Faceless) {
    match faceless {}
  }

  /// Never called: [`Faceless`] has no value.
  fn with_rgb_u16(self, faceless: Faceless) -> Self
  where
    Self: Sized,
  {
    match faceless {}
  }
}

impl<T: ?Sized> FacelessSink for T {}

/// Seals [`Source`] and [`FromSpec`]: only the crate's own types implement
/// them, so neither the frame ↔ [`Convert`] contract nor the set of
/// spec-derivable walk options is user-extensible.
pub(crate) mod sealed {
  /// The sealing supertrait of [`Source`](super::Source).
  pub trait Sealed {}

  /// The sealing supertrait of [`FromSpec`](super::FromSpec).
  pub trait SealedOptions {}
}

/// Derives a source family's walk [`Options`](Source::Options) from the single
/// colorimetric truth of a [`ColorSpec`].
///
/// This is what lets [`Convert::spec`] seed the per-format knobs from the same
/// spec that configures the sink, so the range is never passed twice. The
/// [`Default`] supertrait supplies the no-spec fallback (the
/// unspecified-colorimetry defaults each `Options` type already defines).
///
/// Since mediaframe 0.4 the colour *selector* is not among these knobs — the
/// walkers ask the sink for it — so what a spec seeds here is the quantisation
/// range and nothing else.
///
/// Not every `Options` type can implement this: a value that is *frame
/// intrinsic* rather than colorimetric — the Bayer mosaic pattern of
/// [`BayerOptions`](crate::walker::BayerOptions), which has no [`Default`]
/// because it cannot be guessed — is excluded by construction.
///
/// Sealed: only the crate's own walk-option types ([`YuvOptions`] and the
/// knob-free `()`) implement it, so the set of spec-derivable options is not
/// user-extensible.
pub trait FromSpec: Default + sealed::SealedOptions {
  /// Builds the walk options carried by `spec`, defaulting any knob the spec
  /// does not describe.
  ///
  /// # Errors
  ///
  /// Returns [`UnsupportedKernelMatrixError`](crate::UnsupportedKernelMatrixError)
  /// when the spec names a colour matrix the implementor has no decode for —
  /// see [`ColorSpec::kernel_matrix`](crate::ColorSpec::kernel_matrix).
  fn from_spec(spec: &ColorSpec) -> Result<Self, crate::UnsupportedKernelMatrixError>;
}

impl sealed::SealedOptions for YuvOptions {}
impl FromSpec for YuvOptions {
  /// The spec's resolved range becomes the YUV walk options; everything
  /// colorimetric — the matrix included, since mediaframe 0.4 — is
  /// sink-consumed, not walker-consumed.
  #[inline]
  fn from_spec(spec: &ColorSpec) -> Result<Self, crate::UnsupportedKernelMatrixError> {
    Ok(Self::from_color_spec(spec))
  }
}

impl sealed::SealedOptions for () {}
impl FromSpec for () {
  /// The knob-free sources (XYZ12, a frame-intrinsic palette / conversion)
  /// have no spec-derived state.
  #[inline]
  fn from_spec(_spec: &ColorSpec) -> Result<Self, crate::UnsupportedKernelMatrixError> {
    Ok(())
  }
}

/// A validated source-frame borrow that [`Convert`] can decode.
///
/// Implemented, per source format, by the crate's `walker!` table — mapping the
/// frame borrow (`Self`) to its zero-sized [`Marker`](Self::Marker), its walk
/// [`Options`](Self::Options), and the underlying `{fmt}_to` walker. Sealed: the
/// contract is not user-implementable.
///
/// Every sinker-supported source format implements `Source` **except** the
/// Bayer CFA sources (`Bayer` / `Bayer16`): their
/// [`BayerOptions`](crate::walker::BayerOptions) has no [`Default`] (the mosaic
/// pattern is frame-intrinsic, not colorimetric), so it cannot be a
/// [`FromSpec`] [`Options`](Self::Options). Decode Bayer through the
/// [`Walker`](crate::Walker) / [`MixedSinker`] tier instead — it is Tier-1-only
/// by design.
pub trait Source: sealed::Sealed {
  /// The zero-sized source-format marker (the `F` on
  /// [`MixedSinker`]).
  type Marker: SourceFormat;

  /// The per-format walk options
  /// ([`YuvOptions`], …), derivable from a
  /// [`ColorSpec`] via [`FromSpec`].
  type Options: FromSpec;

  /// The frame's logical `(width, height)` in pixels.
  fn dimensions(&self) -> (usize, usize);

  /// Walks the frame into an assembled `sink`. Called by [`Convert::run`]
  /// after the sink's outputs and tweaks are attached; delegates to the
  /// format's `{fmt}_to` walker.
  #[doc(hidden)]
  fn walk_mixed<R>(
    &self,
    opts: &Self::Options,
    sink: &mut MixedSinker<'_, Self::Marker, R>,
  ) -> Result<(), MixedSinkerError>
  where
    R: Resampler;

  /// Attaches a packed RGBA output buffer to `sink`. Separate from the
  /// generic-over-format `set_rgb` / `set_luma` because RGBA is a per-format
  /// sink capability; called by [`Convert::run`] only when
  /// [`Convert::rgba`] was set.
  #[doc(hidden)]
  fn attach_rgba<'sink, R>(
    sink: &mut MixedSinker<'sink, Self::Marker, R>,
    buf: &'sink mut [u8],
  ) -> Result<(), MixedSinkerError>
  where
    R: Resampler;

  /// What [`Convert::rgb_u16`]'s buffer is held as on this source: the buffer
  /// itself where the source has a `u16` RGB face ([`RgbU16Source`]), and the
  /// uninhabited [`Infallible`](core::convert::Infallible) where it has none.
  /// A faceless source can therefore never hold one, and
  /// [`attach_rgb_u16`](Self::attach_rgb_u16) has no case to refuse.
  #[doc(hidden)]
  type RgbU16Slot<'b>;

  /// Attaches the `u16` RGB buffer [`Convert::rgb_u16`] was handed to `sink`
  /// — the sink tier's `set_rgb_u16` where the face exists, and an empty
  /// `match` where the slot cannot hold one. Called by [`Convert::run`] only
  /// when [`Convert::rgb_u16`] was set.
  #[doc(hidden)]
  fn attach_rgb_u16<'sink, R>(
    sink: &mut MixedSinker<'sink, Self::Marker, R>,
    slot: Self::RgbU16Slot<'sink>,
  ) -> Result<(), MixedSinkerError>
  where
    R: Resampler;
}

/// The bit depth of each channel of a source's packed `u16` RGB face
/// ([`RgbU16Source::RGB_U16_DEPTHS`]), in the face's own `R, G, B` order: each
/// sample of a channel lies in `0..=(1 << depth) - 1`.
///
/// Every source's face has one depth on all three channels but eight legacy
/// packed RGB formats', whose channels keep their own widths: `Rgb565` and
/// `Bgr565` (5, 6, 5), `Rgb8` and `Bgr8` (3, 3, 2), and `Rgb4`, `Bgr4`,
/// `Rgb4Byte` and `Bgr4Byte` (1, 2, 1). [`uniform`](Self::uniform) is the one
/// depth where there is one — the `bits`
/// [`SdrMapping::map_rgb16`](crate::hdr::SdrMapping::map_rgb16) takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RgbDepths {
  red: u32,
  green: u32,
  blue: u32,
}

impl RgbDepths {
  /// The depths of a face, channel by channel.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[must_use]
  pub const fn new(red: u32, green: u32, blue: u32) -> Self {
    Self { red, green, blue }
  }

  /// The depths of a face of `depth` on every channel.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[must_use]
  pub const fn splat(depth: u32) -> Self {
    Self::new(depth, depth, depth)
  }

  /// The red channel's depth.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn red(self) -> u32 {
    self.red
  }

  /// The green channel's depth.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn green(self) -> u32 {
    self.green
  }

  /// The blue channel's depth.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn blue(self) -> u32 {
    self.blue
  }

  /// The one depth of all three channels where they agree, and [`None`] for a
  /// face whose channels differ.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[must_use]
  pub const fn uniform(self) -> Option<u32> {
    if self.red == self.green && self.green == self.blue {
      Some(self.red)
    } else {
      None
    }
  }
}

/// A [`Source`] whose decode has a packed `u16` RGB face: what
/// [`Convert::rgb_u16`] writes, the same face as the sink tier's
/// `MixedSinker::with_rgb_u16`.
///
/// Every sample of a channel lies in `0..=(1 << depth) - 1`, `depth` that
/// channel's in [`RGB_U16_DEPTHS`](Self::RGB_U16_DEPTHS). The depths are
/// stated once, by the source's own row of the `walker!` table:
///
/// - an integer source carries its NATIVE depth on every channel — a 10-bit
///   format's samples in `0..=1023`, low-packed, as FFmpeg's `yuv420p10le`
///   stores them (a `p010` consumer shifts them left by 6);
/// - a source the face widens or narrows to the full `u16` range carries 16:
///   `Pal8` (`(x << 8) | x`), the 32-bit integer formats (`>> 16`), the float
///   formats (clamped to `[0, 1]`, `× 65535`) and XYZ12 (linear `× 65535`);
/// - the 1-bit and 8-bit gray sources carry 8 (zero-extended);
/// - a legacy packed RGB source carries each channel's own width, unexpanded:
///   5 for `Rgb555` and `Bgr555`, 4 for `Rgb444` and `Bgr444`, and, channel
///   by channel, 5/6/5 for `Rgb565` and `Bgr565`, 3/3/2 for `Rgb8` and
///   `Bgr8`, and 1/2/1 for `Rgb4`, `Bgr4`, `Rgb4Byte` and `Bgr4Byte`.
///
/// Where the three agree ([`RgbDepths::uniform`]), the depth is the `bits`
/// that [`SdrMapping::map_rgb16`](crate::hdr::SdrMapping::map_rgb16) takes, so
/// an HDR picture can be decoded and mapped at the precision its source has.
///
/// Every source whose sink tier has `with_rgb_u16` implements this trait: a
/// `walker!` row that states no face over a sink that has one does not build.
/// A source with no `u16` face — every 8-bit YUV, RGB, YUVA and GBR source —
/// does not implement it, so [`Convert::rgb_u16`] on it is a compile error, as
/// `with_rgb_u16` is on its sink. Sealed through [`Source`].
pub trait RgbU16Source: Source {
  /// The bit depth of each channel of the face.
  const RGB_U16_DEPTHS: RgbDepths;

  /// `buf` as this source's [`RgbU16Slot`](Source::RgbU16Slot).
  #[doc(hidden)]
  fn rgb_u16_slot(buf: &mut [u16]) -> Self::RgbU16Slot<'_>;
}

/// The golden one-call decode builder — Tier 0 of pixon's API.
///
/// Construct with [`Convert::from`]`(&frame)`, attach outputs and (optional)
/// colorimetry / resize / tuning with the infallible consuming setters, then
/// [`run`](Self::run) exactly once. `R` is the resampling strategy, defaulting
/// to the identity ([`NoopResampler`]); [`resize`](Self::resize) /
/// [`resize_with`](Self::resize_with) transition it.
///
/// ```
/// # #[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
/// # fn golden() -> Result<(), pixon::Error> {
/// use pixon::{Convert, ColorMatrix, ColorSpec, DynamicRange, PixelFormat};
/// use pixon::frame::Yuv420pFrame;
///
/// let (w, h) = (8u32, 8u32);
/// let y = [16u8; 64];
/// let (u, v) = ([128u8; 16], [128u8; 16]);
/// let frame = Yuv420pFrame::new(&y, &u, &v, w, h, w, w / 2, w / 2);
/// let spec = ColorSpec::resolve(PixelFormat::Yuv420p, DynamicRange::Limited, ColorMatrix::Bt709);
///
/// let mut rgb = [0u8; 4 * 4 * 3];
/// let mut luma = [0u8; 4 * 4];
/// Convert::from(&frame)
///     .spec(spec)
///     .resize(4, 4)
///     .rgb(&mut rgb)
///     .luma(&mut luma)
///     .run()?;
/// # Ok(())
/// # }
/// # #[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
/// # golden().unwrap();
/// ```
pub struct Convert<'a, Fr: Source, R = NoopResampler> {
  frame: &'a Fr,
  resampler: R,
  rgb: Option<&'a mut [u8]>,
  rgba: Option<&'a mut [u8]>,
  luma: Option<&'a mut [u8]>,
  /// The `u16` RGB buffer, held as the source's slot: `None` for every source
  /// without the face, because its slot type has no value.
  rgb_u16: Option<Fr::RgbU16Slot<'a>>,
  // HSV, the siting-aware chroma tweaks, the ST 428 toggle, and the native
  // fast-tier flag are all sink knobs gated to the source families whose sink
  // exposes them. PR-A implements [`Source`] only for the planar-YUV proof
  // formats, so these ride the matching family gates; a later PR widens them
  // alongside the table.
  #[cfg(feature = "yuv-planar")]
  hsv: Option<(&'a mut [u8], &'a mut [u8], &'a mut [u8])>,
  spec: Option<ColorSpec>,
  format_options: Option<Fr::Options>,
  simd: bool,
  #[cfg(feature = "yuv-planar")]
  chroma_location: Option<ChromaLocation>,
  #[cfg(feature = "yuv-planar")]
  st428: Option<St428Interpretation>,
  #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
  native: Option<bool>,
  /// The XYZ12 output gamut, set by [`Convert::target_gamut`] — which only
  /// exists on an XYZ12 builder, so this is always `None` for every other
  /// source. Applied to the sink in [`Convert::run`], where the open
  /// descriptor is exchanged for the closed selector.
  #[cfg(feature = "xyz")]
  target_gamut: Option<crate::DcpTargetGamut>,
}

impl<'a, Fr: Source> From<&'a Fr> for Convert<'a, Fr, NoopResampler> {
  /// Captures the frame; dimensions are read from it at [`run`](Convert::run).
  #[inline]
  fn from(frame: &'a Fr) -> Self {
    Convert {
      frame,
      resampler: NoopResampler,
      rgb: None,
      rgba: None,
      luma: None,
      rgb_u16: None,
      #[cfg(feature = "yuv-planar")]
      hsv: None,
      spec: None,
      format_options: None,
      simd: true,
      #[cfg(feature = "yuv-planar")]
      chroma_location: None,
      #[cfg(feature = "yuv-planar")]
      st428: None,
      #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
      native: None,
      #[cfg(feature = "xyz")]
      target_gamut: None,
    }
  }
}

impl<'a, Fr: Source, R> Convert<'a, Fr, R> {
  /// Moves every non-resampler field onto a builder carrying a new resampler
  /// strategy — the shared body of [`resize`](Self::resize) /
  /// [`resize_with`](Self::resize_with).
  #[inline]
  fn rebind_resampler<R2>(self, resampler: R2) -> Convert<'a, Fr, R2> {
    Convert {
      frame: self.frame,
      resampler,
      rgb: self.rgb,
      rgba: self.rgba,
      luma: self.luma,
      rgb_u16: self.rgb_u16,
      #[cfg(feature = "yuv-planar")]
      hsv: self.hsv,
      spec: self.spec,
      format_options: self.format_options,
      simd: self.simd,
      #[cfg(feature = "yuv-planar")]
      chroma_location: self.chroma_location,
      #[cfg(feature = "yuv-planar")]
      st428: self.st428,
      #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
      native: self.native,
      #[cfg(feature = "xyz")]
      target_gamut: self.target_gamut,
    }
  }

  /// Sets the single colorimetric truth. Configures the whole decode — the
  /// matrix the rows are stamped with, the siting, the primaries, the
  /// transfer and the non-affine tag all go to the sink — and seeds the
  /// walk [`Options`](Source::Options)' quantisation range via [`FromSpec`]
  /// (unless overridden by [`format_options`](Self::format_options)).
  #[must_use]
  #[inline]
  pub fn spec(mut self, spec: ColorSpec) -> Self {
    self.spec = Some(spec);
    self
  }

  /// Overrides the per-format walk options that [`spec`](Self::spec) would
  /// otherwise derive — the rare case where a knob is known out of band.
  ///
  /// Since mediaframe 0.4 the options carry no colour value at all (the
  /// walkers ask the sink), so this can no longer put a different matrix in
  /// front of each reader: what it overrides is the quantisation range and
  /// the frame-intrinsic knobs. The single-source rule is total here.
  #[must_use]
  #[inline]
  pub fn format_options(mut self, options: Fr::Options) -> Self {
    self.format_options = Some(options);
    self
  }

  /// Attaches a packed 24-bit RGB output buffer
  /// (`out_width * out_height * 3` bytes).
  #[must_use]
  #[inline]
  pub fn rgb(mut self, buf: &'a mut [u8]) -> Self {
    self.rgb = Some(buf);
    self
  }

  /// Attaches a packed 32-bit RGBA output buffer
  /// (`out_width * out_height * 4` bytes). Alpha is opaque when the source
  /// carries none.
  #[must_use]
  #[inline]
  pub fn rgba(mut self, buf: &'a mut [u8]) -> Self {
    self.rgba = Some(buf);
    self
  }

  /// Attaches a single-plane luma output buffer
  /// (`out_width * out_height` bytes).
  #[must_use]
  #[inline]
  pub fn luma(mut self, buf: &'a mut [u8]) -> Self {
    self.luma = Some(buf);
    self
  }

  /// Attaches three HSV output planes (`out_width * out_height` bytes each).
  #[cfg(feature = "yuv-planar")]
  #[must_use]
  #[inline]
  pub fn hsv(mut self, h: &'a mut [u8], s: &'a mut [u8], v: &'a mut [u8]) -> Self {
    self.hsv = Some((h, s, v));
    self
  }

  /// Sets whether row primitives dispatch to their SIMD backend (default
  /// `true`); pass `false` to force the scalar reference path.
  #[must_use]
  #[inline]
  pub fn simd(mut self, simd: bool) -> Self {
    self.simd = simd;
    self
  }

  /// Pins the source [`ChromaLocation`] for siting-aware 4:2:0 chroma
  /// reconstruction, overriding the value [`spec`](Self::spec) carries.
  #[cfg(feature = "yuv-planar")]
  #[must_use]
  #[inline]
  pub fn chroma_location(mut self, loc: ChromaLocation) -> Self {
    self.chroma_location = Some(loc);
    self
  }

  /// Selects how [`Primaries::SmpteSt428`](crate::Primaries::SmpteSt428) is
  /// interpreted for the `ChromaDerivedNcl` decode (see
  /// [`St428Interpretation`]).
  #[cfg(feature = "yuv-planar")]
  #[must_use]
  #[inline]
  pub fn st428_interpretation(mut self, interp: St428Interpretation) -> Self {
    self.st428 = Some(interp);
    self
  }

  /// Toggles the native (source-domain averaging) fast tier for area
  /// downscales (default `true` on the sink).
  #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
  #[must_use]
  #[inline]
  pub fn native(mut self, native: bool) -> Self {
    self.native = Some(native);
    self
  }
}

/// The `u16` RGB face, which exists only on a builder over an
/// [`RgbU16Source`].
///
/// Like the XYZ12 gamut below, it is not a setter on the generic builder: a
/// source with no such face has nothing to write into it, and a buffer it
/// silently left stale is the failure mode this seam exists to remove.
impl<'a, Fr: RgbU16Source, R> Convert<'a, Fr, R> {
  /// Attaches a packed `u16` RGB output buffer (`out_width * out_height * 3`
  /// elements), each channel's samples at the source's depth for it
  /// ([`RGB_U16_DEPTHS`](RgbU16Source::RGB_U16_DEPTHS)).
  ///
  /// The face is the sink tier's `with_rgb_u16`, assembled for you: the same
  /// samples, byte for byte, at native size and through
  /// [`resize`](Convert::resize) / [`resize_with`](Convert::resize_with).
  /// It combines with every other output.
  ///
  /// ```
  /// # #[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
  /// # fn native() -> Result<(), pixon::Error> {
  /// use pixon::{ColorMatrix, ColorSpec, Convert, DynamicRange, PixelFormat, RgbDepths, RgbU16Source};
  /// use pixon::frame::Yuv420p10Frame;
  ///
  /// let (w, h) = (4u32, 2u32);
  /// let y = [940u16; 8]; // limited-range 10-bit white
  /// let (u, v) = ([512u16; 2], [512u16; 2]);
  /// let frame = Yuv420p10Frame::new(&y, &u, &v, w, h, w, w / 2, w / 2);
  /// let spec = ColorSpec::resolve(PixelFormat::Yuv420p10Le, DynamicRange::Limited, ColorMatrix::Bt2020Ncl);
  ///
  /// let mut rgb = [0u16; 4 * 2 * 3];
  /// Convert::from(&frame).spec(spec).rgb_u16(&mut rgb).run()?;
  /// assert_eq!(<Yuv420p10Frame<'_> as RgbU16Source>::RGB_U16_DEPTHS, RgbDepths::splat(10));
  /// assert!(rgb.iter().all(|&sample| sample == 1023));
  /// # Ok(())
  /// # }
  /// # #[cfg(all(any(feature = "std", feature = "alloc"), feature = "yuv-planar"))]
  /// # native().unwrap();
  /// ```
  #[must_use]
  #[inline]
  pub fn rgb_u16(mut self, buf: &'a mut [u16]) -> Self {
    self.rgb_u16 = Some(Fr::rgb_u16_slot(buf));
    self
  }
}

/// The XYZ12 output-gamut door, which exists only on an XYZ12 builder.
///
/// A target gamut is meaningless for every other source, so it is not a
/// setter on the generic builder — a knob that silently does nothing is the
/// failure mode this whole seam exists to remove. Through mediaframe 0.3 the
/// gamut rode `Xyz12Options` and reached the walker; 0.4 asks the sink for
/// it, and this is how [`Convert`] routes it there.
#[cfg(all(feature = "xyz", any(feature = "std", feature = "alloc")))]
impl<const BE: bool, R> Convert<'_, crate::frame::Xyz12Frame<'_, BE>, R> {
  /// Names the RGB gamut the XYZ12 decode converts into.
  ///
  /// The open [`DcpTargetGamut`](crate::DcpTargetGamut) descriptor is
  /// exchanged for the closed selector in [`run`](Self::run), so a gamut this
  /// build has no luma basis for fails the whole convert before a row is
  /// read. Default: [`DcpTargetGamut`]'s own — DCI-P3, the SMPTE ST 428-1
  /// D-Cinema decode target.
  ///
  /// [`DcpTargetGamut`]: crate::DcpTargetGamut
  #[must_use]
  #[inline]
  pub fn target_gamut(mut self, gamut: crate::DcpTargetGamut) -> Self {
    self.target_gamut = Some(gamut);
    self
  }
}

impl<'a, Fr: Source> Convert<'a, Fr, NoopResampler> {
  /// Area (box-coverage) downscale to `out_width * out_height` — the
  /// `cv2.INTER_AREA` convention. [`run`](Convert::run) errors on upscale,
  /// matching [`AreaResampler`].
  #[must_use]
  #[inline]
  pub fn resize(self, out_width: usize, out_height: usize) -> Convert<'a, Fr, AreaResampler> {
    self.rebind_resampler(AreaResampler::to(out_width, out_height))
  }

  /// Filtered resample to `out_width * out_height` under `kernel` (any ratio) —
  /// the windowed-filter (PIL) convention. See [`FilteredResampler`].
  #[must_use]
  #[inline]
  pub fn resize_with<K: FilterKernel>(
    self,
    kernel: K,
    out_width: usize,
    out_height: usize,
  ) -> Convert<'a, Fr, FilteredResampler<K>> {
    self.rebind_resampler(FilteredResampler::new(out_width, out_height, kernel))
  }
}

impl<Fr: Source, R: Resampler> Convert<'_, Fr, R> {
  /// Assembles the sink, attaches the requested outputs and tweaks, resolves
  /// the walk options, and decodes the frame. The only fallible call.
  ///
  /// Options resolve as [`format_options`](Self::format_options), else
  /// [`FromSpec::from_spec`] of the [`spec`](Self::spec), else
  /// [`Default`].
  ///
  /// # Errors
  ///
  /// [`MixedSinkerError`] — a rejected resize plan (e.g. an upscale requested
  /// through [`resize`](Self::resize)), a short output buffer, or a walk-time
  /// failure.
  pub fn run(self) -> Result<(), MixedSinkerError> {
    let (w, h) = self.frame.dimensions();
    let mut sink = MixedSinker::<Fr::Marker, R>::with_resampler(w, h, self.resampler)?;

    if let Some(buf) = self.rgb {
      sink.set_rgb(buf)?;
    }
    if let Some(buf) = self.luma {
      sink.set_luma(buf)?;
    }
    #[cfg(feature = "yuv-planar")]
    if let Some((h_plane, s_plane, v_plane)) = self.hsv {
      sink.set_hsv(h_plane, s_plane, v_plane)?;
    }
    if let Some(buf) = self.rgba {
      Fr::attach_rgba(&mut sink, buf)?;
    }
    if let Some(slot) = self.rgb_u16 {
      Fr::attach_rgb_u16(&mut sink, slot)?;
    }

    sink.set_simd(self.simd);
    // The spec's matrix is exchanged for the closed kernel vocabulary here,
    // at the door: a matrix pixon has no decode for fails the whole convert
    // before a row is read, rather than decoding as BT.709. Since
    // mediaframe 0.4 the sink is also what the walk reads the matrix *from*,
    // so this one call configures both halves.
    if let Some(spec) = self.spec.as_ref() {
      sink.set_color_spec(spec)?;
    }
    #[cfg(feature = "xyz")]
    if let Some(gamut) = self.target_gamut.as_ref() {
      sink.set_target_gamut(gamut)?;
    }
    #[cfg(feature = "yuv-planar")]
    {
      if let Some(loc) = self.chroma_location {
        sink.set_chroma_location(loc.clone());
      }
      if let Some(interp) = self.st428 {
        sink.set_st428_interpretation(interp);
      }
    }
    #[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
    if let Some(native) = self.native {
      sink.set_native(native);
    }

    let options = match self.format_options {
      Some(options) => options,
      None => match self.spec.as_ref() {
        Some(spec) => Fr::Options::from_spec(spec)?,
        None => Fr::Options::default(),
      },
    };
    self.frame.walk_mixed(&options, &mut sink)
  }
}
