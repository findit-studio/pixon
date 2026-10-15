//! The BT.2100 curves and the analysis mapping, held to the published values.
//!
//! The curve anchors are the ones the private `pq_hlg` module carried until
//! the curves became public: `colour-science` docstring examples
//! (`eotf_ST2084` / `eotf_inverse_ST2084`, `oetf_ARIBSTDB67`) and the exact
//! construction points of ITU-R BT.2100-2 Tables 4 and 5. A transcription
//! error in any constant moves them well outside the stated tolerances. The
//! mapping laws state the numbers the module note states: BT.2408's
//! reference white and 18% grey, BT.2087's gamut matrix, the knee at 0.75,
//! and the inverse BT.1886 encode.

use super::{
  HLG_NOMINAL_PEAK_NITS, HdrTransfer, KNEE_START, MapError, PQ_PEAK_NITS, REFERENCE_WHITE_NITS,
  SdrMapping, expf32, hlg_inverse_oetf, hlg_oetf, hlg_ootf, hlg_system_gamma, mapping::knee,
  powf32, pq_eotf, pq_inverse_eotf,
};
use crate::{Primaries, Transfer};

/// BT.2020's luminance row, BT.2100 Table 5's `Y_s` weights.
const BT2020_LUMINANCE: [f32; 3] = [0.2627, 0.6780, 0.0593];

/// The knee the module note states. The laws spell the number rather than
/// read [`KNEE_START`], so a moved knee is caught here.
const STATED_KNEE: f32 = 0.75;

/// The stated knee applied to light relative to reference white.
fn stated_knee(light: f32) -> f32 {
  if light <= STATED_KNEE {
    light
  } else {
    let room = 1.0 - STATED_KNEE;
    STATED_KNEE + room * (1.0 - expf32(-(light - STATED_KNEE) / room))
  }
}

/// A `[0, 1]` signal to a full-range 8-bit code, rounded to nearest.
fn code(signal: f32) -> u8 {
  (signal * 255.0 + 0.5) as u8
}

/// Every HDR transfer, as the vocabulary spells it.
const HDR: [Transfer; 2] = [Transfer::SmpteSt2084Pq, Transfer::AribStdB67Hlg];

// ---- the BT.2100 curves ---------------------------------------------------

#[test]
fn pq_matches_st2084_reference() {
  // BT.2100 anchors: signal 1.0 ↔ linear 1.0 (= 10 000 cd/m²), signal 0 ↔
  // linear 0. The signal-1.0 fixed point holds iff c1 = c3 − c2 + 1.
  assert!(
    (pq_eotf(1.0) - 1.0).abs() <= 1e-6,
    "eotf(1.0)={}",
    pq_eotf(1.0)
  );
  assert!(pq_eotf(0.0).abs() <= 1e-9, "eotf(0.0)={}", pq_eotf(0.0));
  assert!(
    (pq_inverse_eotf(1.0) - 1.0).abs() <= 1e-5,
    "oetf(1.0)={}",
    pq_inverse_eotf(1.0)
  );

  // colour-science `eotf_inverse_ST2084(100)` = 0.5080784: 100 cd/m² is the
  // normalised linear 0.01, encoded to signal 0.5080784. This single point
  // pins all five PQ constants in the encode direction.
  const SIGNAL_100_NITS: f32 = 0.508_078_4;
  assert!(
    (pq_inverse_eotf(0.01) - SIGNAL_100_NITS).abs() <= 5e-4,
    "oetf(0.01)={} want {SIGNAL_100_NITS}",
    pq_inverse_eotf(0.01),
  );
  // colour-science `eotf_ST2084(0.508078421517399)` = 100 cd/m²: the decode
  // direction of the same anchor (normalised 0.01).
  assert!(
    (pq_eotf(SIGNAL_100_NITS) - 0.01).abs() <= 1e-4,
    "eotf({SIGNAL_100_NITS})={} want 0.01",
    pq_eotf(SIGNAL_100_NITS),
  );

  // PQ's inverse-EOTF maps linear 0 to a small non-zero signal `c1^m2`
  // (≈ 7.3e-7), a known property of the curve — assert it stays negligible.
  assert!(
    pq_inverse_eotf(0.0).abs() <= 1e-5,
    "oetf(0.0)={}",
    pq_inverse_eotf(0.0)
  );
}

#[test]
fn a_super_white_saturates_at_the_peak() {
  // PQ signal E' is defined on [0, 1]; a super-white magnitude saturates at
  // the peak (linear 1.0) — it must NOT cross the den=0 pole at |c| ~= 1.99
  // (which overflows toward +inf just below it and folds to black just above
  // it). Pins values straddling the pole.
  for &c in &[1.0_f32, 1.5, 1.99, 2.0, 10.0] {
    let y = pq_eotf(c);
    assert!(y.is_finite(), "pq_eotf({c})={y} must be finite");
    assert!(
      (y - 1.0).abs() <= 1e-6,
      "pq_eotf({c})={y} must saturate at 1.0 (not inf/black)"
    );
  }
  // Odd extension: super-black saturates at -1.0.
  assert!(
    (pq_eotf(-2.0) + 1.0).abs() <= 1e-6,
    "pq_eotf(-2.0)={}",
    pq_eotf(-2.0)
  );

  // HLG likewise clamps the [0, 1] signal domain (its log segment would grow
  // unbounded for E' > 1); a super-white input equals the peak hlg_inverse_oetf(1.0).
  let hlg_peak = hlg_inverse_oetf(1.0);
  for &c in &[1.0_f32, 2.0, 10.0] {
    let e = hlg_inverse_oetf(c);
    assert!(
      e.is_finite() && (e - hlg_peak).abs() <= 1e-6,
      "hlg_inverse_oetf({c})={e} must saturate at hlg_inverse_oetf(1.0)={hlg_peak}"
    );
  }
}

#[test]
fn hlg_matches_bt2100_reference() {
  // BT.2100 breakpoint: scene-linear 1/12 (HLG reference white) encodes to
  // signal 0.5, and the lower (gamma) segment is exact there.
  assert!(
    (hlg_oetf(1.0 / 12.0) - 0.5).abs() <= 1e-6,
    "oetf(1/12)={}",
    hlg_oetf(1.0 / 12.0),
  );
  // Inverse breakpoint: signal 0.5 decodes back to scene-linear 1/12.
  assert!(
    (hlg_inverse_oetf(0.5) - 1.0 / 12.0).abs() <= 1e-6,
    "eotf(0.5)={}",
    hlg_inverse_oetf(0.5),
  );

  // colour-science `oetf_ARIBSTDB67(0.18)` = 0.2121320. Under the BT.2100
  // normalisation E = E_arib / 12, that is `oetf(0.015)` — a lower (gamma)
  // segment point: sqrt(3·0.015) = 0.2121320.
  assert!(
    (hlg_oetf(0.015) - 0.212_132).abs() <= 1e-5,
    "oetf(0.015)={}",
    hlg_oetf(0.015),
  );
  assert!(
    (hlg_inverse_oetf(0.212_132) - 0.015).abs() <= 1e-5,
    "eotf(0.2121320)={}",
    hlg_inverse_oetf(0.212_132),
  );

  // Upper (log) segment construction anchors — these exercise a, b and c:
  // scene-linear 1.0 ↔ signal 1.0 (c is defined to make this hold), and
  // scene-linear 0 ↔ signal 0.
  assert!(
    (hlg_oetf(1.0) - 1.0).abs() <= 1e-4,
    "oetf(1.0)={}",
    hlg_oetf(1.0)
  );
  assert!(
    (hlg_inverse_oetf(1.0) - 1.0).abs() <= 1e-4,
    "eotf(1.0)={}",
    hlg_inverse_oetf(1.0)
  );
  assert!(hlg_oetf(0.0).abs() <= 1e-9, "oetf(0.0)={}", hlg_oetf(0.0));
  assert!(
    hlg_inverse_oetf(0.0).abs() <= 1e-9,
    "eotf(0.0)={}",
    hlg_inverse_oetf(0.0)
  );
}

#[test]
fn each_curve_inverts_its_pair() {
  // EOTF and OETF are exact analytic inverses; verify the f32 round-trip
  // over the unit signal interval. (The round-trip is insensitive to the
  // constants' absolute values — the `*_matches_*_reference` tests pin
  // those — so this guards only the inverse relationship.)
  for i in 0..=256 {
    let c = i as f32 / 256.0;
    let pq = pq_inverse_eotf(pq_eotf(c));
    assert!((pq - c).abs() <= 2e-3, "pq oetf(eotf({c}))={pq}");
    let pq2 = pq_eotf(pq_inverse_eotf(c));
    assert!((pq2 - c).abs() <= 2e-3, "pq eotf(oetf({c}))={pq2}");
    let hlg = hlg_oetf(hlg_inverse_oetf(c));
    assert!((hlg - c).abs() <= 1e-3, "hlg oetf(eotf({c}))={hlg}");
    let hlg2 = hlg_inverse_oetf(hlg_oetf(c));
    assert!((hlg2 - c).abs() <= 1e-3, "hlg eotf(oetf({c}))={hlg2}");
  }
}

// miri's interpreted floating-point diverges from hardware for the PQ/HLG
// transcendentals past this test's tolerance.
#[cfg_attr(miri, ignore)]
#[test]
fn each_curve_is_odd() {
  // The per-channel curves are odd (`f(-c) == -f(c)`) so super-black
  // excursions linearise symmetrically rather than folding — the same
  // contract the SDR curves honour.
  for &c in &[0.05_f32, 0.25, 0.6, 0.9] {
    assert!(
      (pq_eotf(-c) + pq_eotf(c)).abs() <= 1e-6,
      "pq eotf odd at {c}"
    );
    assert!(
      (pq_inverse_eotf(-c) + pq_inverse_eotf(c)).abs() <= 1e-6,
      "pq oetf odd at {c}"
    );
    assert!(
      (hlg_inverse_oetf(-c) + hlg_inverse_oetf(c)).abs() <= 1e-6,
      "hlg eotf odd at {c}"
    );
    assert!(
      (hlg_oetf(-c) + hlg_oetf(c)).abs() <= 1e-6,
      "hlg oetf odd at {c}"
    );
  }
}

#[test]
fn the_hdr_curves_differ_from_sdr() {
  // A mid-tone linearises differently under the HDR curves than the SDR
  // BT.1886 default, so the BT.2100 decode is observably distinct.
  let c = 0.5_f32;
  let bt1886 = crate::resample::TransferFunction::Bt1886.eotf(c);
  let pq = pq_eotf(c);
  let hlg = hlg_inverse_oetf(c);
  assert!((pq - bt1886).abs() > 1e-3, "PQ vs BT.1886 must differ");
  assert!((hlg - bt1886).abs() > 1e-3, "HLG vs BT.1886 must differ");
  assert!((pq - hlg).abs() > 1e-3, "PQ vs HLG must differ");
}

// ---- BT.2100 and BT.2408 anchors the mapping is built on -------------------

/// BT.2408 puts HDR reference white, 203 cd/m², at PQ 58%, and BT.2100's
/// 1000 cd/m² at PQ 75%.
#[test]
fn pq_places_reference_white_at_58_percent() {
  let white = pq_inverse_eotf(REFERENCE_WHITE_NITS / PQ_PEAK_NITS);
  assert!((white - 0.5806).abs() <= 1e-3, "PQ(203 cd/m²) = {white}");
  let thousand = pq_inverse_eotf(1000.0 / PQ_PEAK_NITS);
  assert!(
    (thousand - 0.7518).abs() <= 1e-3,
    "PQ(1000 cd/m²) = {thousand}"
  );
}

/// BT.2408 on the 1000 cd/m² reference display: HLG 75% is reference white,
/// 203 cd/m², and HLG 38% is 18% grey, 26 cd/m².
#[test]
fn hlg_places_reference_white_at_75_percent() {
  let white = hlg_ootf(
    [hlg_inverse_oetf(0.75); 3],
    BT2020_LUMINANCE,
    HLG_NOMINAL_PEAK_NITS,
  );
  assert!(
    (white[1] - 203.0).abs() <= 1.0,
    "HLG 75% = {} cd/m²",
    white[1]
  );
  let grey = hlg_ootf(
    [hlg_inverse_oetf(0.38); 3],
    BT2020_LUMINANCE,
    HLG_NOMINAL_PEAK_NITS,
  );
  assert!((grey[1] - 26.0).abs() <= 0.5, "HLG 38% = {} cd/m²", grey[1]);
  assert_eq!(
    hlg_ootf([0.0; 3], BT2020_LUMINANCE, HLG_NOMINAL_PEAK_NITS),
    [0.0; 3],
    "black is black on any display"
  );
}

/// BT.2100 Table 5: γ = 1.2 at 1000 cd/m², and `1.2 + 0.42·log10(L_W/1000)`
/// beside it.
#[test]
fn the_hlg_system_gamma_is_bt2100s() {
  assert_eq!(hlg_system_gamma(1000.0), 1.2);
  assert!((hlg_system_gamma(2000.0) - 1.326_433).abs() <= 1e-5);
  assert!((hlg_system_gamma(400.0) - 1.032_867).abs() <= 1e-5);
}

/// The matrix from BT.2020 to BT.709 is ITU-R BT.2087-0's, and BT.2020's
/// luminance row is BT.2100's `Y_s` weights: both derived from the
/// chromaticities `mediaframe` tabulates, not copied.
#[test]
fn the_gamut_matrix_from_bt2020_is_bt2087s() {
  const BT2087: [[f32; 3]; 3] = [
    [1.6605, -0.5876, -0.0728],
    [-0.1246, 1.1329, -0.0083],
    [-0.0182, -0.1006, 1.1187],
  ];
  let mapping = SdrMapping::for_picture(&Transfer::SmpteSt2084Pq, &Primaries::Bt2020)
    .expect("BT.2020 converts");
  let hdr = mapping.hdr.as_ref().expect("PQ maps");
  for (row, (got, want)) in hdr.to_bt709.iter().zip(BT2087).enumerate() {
    for column in 0..3 {
      assert!(
        (got[column] - want[column]).abs() <= 2e-4,
        "[{row}][{column}] = {} want {}",
        got[column],
        want[column],
      );
    }
  }
  for (got, want) in hdr.luminance.iter().zip(BT2020_LUMINANCE) {
    assert!((got - want).abs() <= 1e-4, "luminance {got} want {want}");
  }
}

// ---- the mapping ----------------------------------------------------------

/// The knee starts at 0.75 of SDR white: identity below, slope one at the
/// start, rising above, and never past white. It rises strictly as far as
/// `f32` resolves the increments. Near the asymptote two steps can round to
/// the same value, so beyond two whites the law asks only that it never falls.
#[test]
fn the_knee_starts_where_stated() {
  for step in 0..=75 {
    let light = step as f32 / 100.0;
    assert_eq!(
      knee(light),
      light,
      "below the knee, {light} passes unchanged"
    );
  }
  let slope = (knee(STATED_KNEE + 1e-3) - knee(STATED_KNEE)) / 1e-3;
  assert!((slope - 1.0).abs() <= 1e-2, "slope at the knee {slope}");
  let mut previous = knee(STATED_KNEE);
  for step in 1..=400 {
    let light = STATED_KNEE + step as f32 / 50.0;
    let rolled = knee(light);
    if light <= 2.0 {
      assert!(rolled > previous, "rising at {light}: {rolled}");
    } else {
      assert!(rolled >= previous, "never falling at {light}: {rolled}");
    }
    assert!(
      rolled < light && rolled <= 1.0,
      "compressed at {light}: {rolled}"
    );
    previous = rolled;
  }
  assert!(
    knee(2.0) < 1.0,
    "two whites still below white: {}",
    knee(2.0)
  );
}

/// A PQ ramp and an HLG ramp of neutral greys map into `[0, 1]`, rising,
/// with each output the stated knee of its display light, encoded by the
/// inverse BT.1886 EOTF.
#[test]
fn a_pq_and_an_hlg_ramp_map_into_sdr_with_the_stated_knee() {
  for transfer in HDR {
    let mapping = SdrMapping::for_picture(&transfer, &Primaries::Bt2020).expect("BT.2020 converts");
    let mut previous = 0.0f32;
    for step in 0..=1023 {
      let signal = step as f32 / 1023.0;
      let out = mapping.map_signal([signal; 3]);
      for channel in out {
        assert!(
          (0.0..=1.0).contains(&channel),
          "{transfer:?} {signal}: {out:?}"
        );
        assert!((channel - out[1]).abs() <= 1e-5, "grey stays grey: {out:?}");
      }
      assert!(out[1] + 1e-6 >= previous, "{transfer:?} rising at {signal}");
      previous = out[1];

      let light = match transfer {
        Transfer::SmpteSt2084Pq => pq_eotf(signal) * 10_000.0 / 203.0,
        _ => 1000.0 * powf32(hlg_inverse_oetf(signal), 1.2) / 203.0,
      };
      let shown = powf32(out[1], 2.4);
      let want = stated_knee(light);
      assert!(
        (shown - want).abs() <= 2e-4 + 2e-4 * want,
        "{transfer:?} {signal}: shown {shown} want {want} from light {light}",
      );
      if light > STATED_KNEE {
        assert!(shown < light, "above the knee the light is compressed");
      }
    }
  }
}

/// PQ 58% and HLG 75% (reference white) land at 0.961, and BT.2408's 18%
/// grey (26 cd/m²) at 0.425: the stated knee, then the inverse BT.1886
/// encode.
#[test]
fn reference_white_and_grey_land_where_stated() {
  let pq = SdrMapping::for_picture(&Transfer::SmpteSt2084Pq, &Primaries::Bt2020).expect("PQ");
  let hlg = SdrMapping::for_picture(&Transfer::AribStdB67Hlg, &Primaries::Bt2020).expect("HLG");

  let white = pq.map_signal([pq_inverse_eotf(203.0 / 10_000.0); 3])[1];
  assert!((white - 0.9606).abs() <= 1e-3, "PQ reference white {white}");
  let white = hlg.map_signal([0.75; 3])[1];
  assert!(
    (white - 0.9606).abs() <= 2e-3,
    "HLG reference white {white}"
  );

  let grey = pq.map_signal([pq_inverse_eotf(26.0 / 10_000.0); 3])[1];
  assert!((grey - 0.4247).abs() <= 1e-3, "PQ 18% grey {grey}");
  let grey = hlg.map_signal([0.38; 3])[1];
  assert!((grey - 0.4264).abs() <= 2e-3, "HLG 18% grey {grey}");
}

/// The declared gamut is converted: a saturated BT.2020 green is not the
/// picture the same codes make over BT.709, while a grey is the same in
/// both. An unspecified gamut reads as BT.2020, and BT.709 converts by the
/// identity.
#[test]
fn the_declared_gamut_is_converted_to_bt709() {
  for transfer in HDR {
    let wide = SdrMapping::for_picture(&transfer, &Primaries::Bt2020).expect("BT.2020");
    let narrow = SdrMapping::for_picture(&transfer, &Primaries::Bt709).expect("BT.709");
    let green = [0.3, 0.6, 0.3];
    let (from_wide, from_narrow) = (wide.map_signal(green), narrow.map_signal(green));
    assert!(
      from_wide[0] + 0.05 < from_narrow[0],
      "{transfer:?}: BT.2020 green loses red in BT.709 ({from_wide:?} vs {from_narrow:?})",
    );
    let grey = [0.5; 3];
    let (a, b) = (wide.map_signal(grey), narrow.map_signal(grey));
    for channel in 0..3 {
      assert!(
        (a[channel] - b[channel]).abs() <= 1e-5,
        "grey: {a:?} vs {b:?}"
      );
    }
    assert_eq!(
      SdrMapping::for_picture(&transfer, &Primaries::Unspecified).expect("unspecified"),
      wide,
      "an unspecified gamut is BT.2020",
    );
    let identity = narrow.hdr.as_ref().expect("maps").to_bt709;
    for (row, line) in identity.iter().enumerate() {
      for (column, value) in line.iter().enumerate() {
        let want = if row == column { 1.0 } else { 0.0 };
        assert!(
          (value - want).abs() <= 1e-6,
          "BT.709 to BT.709 [{row}][{column}] {value}"
        );
      }
    }
  }
}

/// Every transfer but PQ and HLG builds a passthrough, and a passthrough
/// leaves a BT.709 picture as it is, through every face.
#[test]
fn a_bt709_picture_passes_through_unchanged() {
  for transfer in [
    Transfer::Bt709,
    Transfer::Unspecified,
    Transfer::Smpte170M,
    Transfer::Bt2020_10Bit,
    Transfer::Iec6196621,
    Transfer::Linear,
  ] {
    let mapping = SdrMapping::for_picture(&transfer, &Primaries::Bt709).expect("SDR");
    assert!(mapping.is_passthrough(), "{transfer:?}");
    assert_eq!(mapping.transfer(), None);
    let original: [u8; 768] = core::array::from_fn(|i| (i * 7 % 256) as u8);
    let mut rgb = original;
    mapping.map_rgb8(&mut rgb);
    assert_eq!(
      rgb, original,
      "{transfer:?}: the 8-bit face changed nothing"
    );
    assert_eq!(mapping.map_signal([0.1, 0.5, 0.9]), [0.1, 0.5, 0.9]);
    let mut narrowed = [0u8; 6];
    mapping
      .map_rgb16(&[0, 512, 1023, 1023, 0, 256], 10, &mut narrowed)
      .expect("whole pixels");
    assert_eq!(
      narrowed,
      [0, 128, 255, 255, 0, 64],
      "only the depth narrows"
    );
  }
  assert!(
    SdrMapping::for_picture(&Transfer::Bt709, &Primaries::Film)
      .expect("an SDR picture is never refused")
      .is_passthrough()
  );
  assert!(SdrMapping::passthrough().is_passthrough());
}

/// A PQ or HLG picture over a gamut with no derivable BT.709 conversion is
/// refused, naming both its transfer and its gamut.
#[test]
fn a_gamut_with_no_conversion_is_refused_by_name() {
  for primaries in [
    Primaries::Film,
    Primaries::Bt470M,
    Primaries::SmpteRp431,
    Primaries::SmpteSt428,
    Primaries::other("vendor-gamut"),
  ] {
    for transfer in HDR {
      let refused = SdrMapping::for_picture(&transfer, &primaries).expect_err("refused");
      assert_eq!(refused.primaries(), &primaries);
      assert_eq!(Some(refused.transfer()), HdrTransfer::of(&transfer));
      let said = std::format!("{refused}");
      assert!(said.contains(primaries.as_str()), "{said}");
      assert!(said.contains(refused.transfer().as_str()), "{said}");
    }
  }
  for primaries in [
    Primaries::Bt709,
    Primaries::Bt2020,
    Primaries::SmpteEg432,
    Primaries::Bt470Bg,
    Primaries::Smpte170M,
    Primaries::Smpte240M,
    Primaries::Ebu3213E,
    Primaries::Unspecified,
  ] {
    for transfer in HDR {
      assert!(
        SdrMapping::for_picture(&transfer, &primaries).is_ok(),
        "{primaries:?} has a D65 white and converts",
      );
    }
  }
}

/// The 8-bit face is the signal face at `code / 255`, quantised to the
/// nearest code.
#[test]
fn the_8_bit_face_is_the_signal_face_quantised() {
  for transfer in HDR {
    for primaries in [Primaries::Bt2020, Primaries::Bt709] {
      let mapping = SdrMapping::for_picture(&transfer, &primaries).expect("converts");
      for value in 0..=255u8 {
        for pixel in [
          [value, value, value],
          [value, 255 - value, value / 2],
          [255, value, 0],
        ] {
          let mut mapped = pixel;
          mapping.map_rgb8(&mut mapped);
          let want = mapping
            .map_signal(pixel.map(|c| c as f32 / 255.0))
            .map(code);
          assert_eq!(mapped, want, "{transfer:?} over {primaries:?}: {pixel:?}");
        }
      }
      let mut partial = [200u8, 100, 50, 25];
      mapping.map_rgb8(&mut partial);
      assert_eq!(partial[3], 25, "a trailing partial pixel is left alone");
    }
  }
}

/// The 16-bit face maps at its native depth, reads a sample above full
/// scale as full scale, and refuses a depth or a length it cannot map.
#[test]
fn the_16_bit_face_maps_at_its_native_depth() {
  let mapping = SdrMapping::for_picture(&Transfer::AribStdB67Hlg, &Primaries::Bt2020).expect("HLG");
  let rgb: [u16; 6] = [0, 600, 1023, 512, 512, 512];
  let mut out = [0u8; 6];
  mapping.map_rgb16(&rgb, 10, &mut out).expect("whole pixels");
  for (pixel, mapped) in rgb.as_chunks::<3>().0.iter().zip(out.as_chunks::<3>().0) {
    let want = mapping
      .map_signal([0, 1, 2].map(|i| f32::from(pixel[i]) / 1023.0))
      .map(code);
    assert_eq!(*mapped, want, "{pixel:?}");
  }

  let (mut over, mut full) = ([0u8; 3], [0u8; 3]);
  mapping.map_rgb16(&[2000; 3], 10, &mut over).expect("whole");
  mapping.map_rgb16(&[1023; 3], 10, &mut full).expect("whole");
  assert_eq!(over, full, "above full scale reads as full scale");

  assert_eq!(
    mapping.map_rgb16(&rgb, 0, &mut out),
    Err(MapError::Depth { bits: 0 })
  );
  assert_eq!(
    mapping.map_rgb16(&rgb, 17, &mut out),
    Err(MapError::Depth { bits: 17 })
  );
  assert_eq!(
    mapping.map_rgb16(&rgb, 10, &mut out[..3]),
    Err(MapError::LengthMismatch {
      samples: 6,
      bytes: 3
    }),
  );
  assert_eq!(
    mapping.map_rgb16(&rgb[..5], 10, &mut out[..5]),
    Err(MapError::LengthMismatch {
      samples: 5,
      bytes: 5
    }),
  );
}

/// The faces never answer a value outside `[0, 1]` or a NaN, whatever
/// they are handed.
#[test]
fn every_output_is_a_signal() {
  for transfer in HDR {
    let mapping = SdrMapping::for_picture(&transfer, &Primaries::Bt2020).expect("converts");
    for input in [
      [-1.0, 0.5, 2.0],
      [f32::NAN, 0.25, 0.75],
      [f32::INFINITY, f32::NEG_INFINITY, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0],
    ] {
      for channel in mapping.map_signal(input) {
        assert!(
          (0.0..=1.0).contains(&channel),
          "{transfer:?} {input:?}: {channel}"
        );
      }
    }
  }
}

/// The constants the module note states are the ones the mapping uses.
#[test]
fn the_stated_constants() {
  assert_eq!(KNEE_START, STATED_KNEE);
  assert_eq!(REFERENCE_WHITE_NITS, 203.0);
  assert_eq!(HLG_NOMINAL_PEAK_NITS, 1000.0);
  assert_eq!(PQ_PEAK_NITS, 10_000.0);
  assert!((hlg_oetf(1.0 / 12.0) - 0.5).abs() <= 1e-6);
}
