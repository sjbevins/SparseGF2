"""Tests for the SampleRecord union-shape contract."""

from __future__ import annotations

import dataclasses

from sparsegf2.circuits import Picture, SampleRecord


def test_required_and_optional_fields():
    fields = {f.name: f for f in dataclasses.fields(SampleRecord)}
    # required (no default)
    required = {
        "sample_seed",
        "picture",
        "total_layers",
        "total_gates",
        "total_measurements",
        "gate_to_meas_ratio_expected",
        "gate_to_meas_ratio_actual",
    }
    for name in required:
        assert fields[name].default is dataclasses.MISSING, f"{name} should be required"
    # optional observables default to None / 0.0
    assert fields["code_dimension"].default is None
    assert fields["entropy_half_cut"].default is None
    assert fields["final_tableau"].default is None
    assert fields["runtime_total_s"].default == 0.0


def test_union_shape_minimal_construction():
    # A record can be built with only the required fields; observables stay None.
    rec = SampleRecord(
        sample_seed=0,
        picture=Picture.PURE_STATE,
        total_layers=10,
        total_gates=40,
        total_measurements=12,
        gate_to_meas_ratio_expected=3.0,
        gate_to_meas_ratio_actual=3.33,
    )
    assert rec.code_dimension is None
    assert rec.entropy_half_cut is None


def test_checkpoint_fields_preserve_preexisting_positional_layout():
    # Before checkpoint support, ``analyses`` immediately followed
    # ``final_tableau``. Additive record fields must not silently reinterpret an
    # older positional constructor call.
    analyses = {"legacy": 7}
    rec = SampleRecord(
        0,
        Picture.PURE_STATE,
        1,
        2,
        3,
        4.0,
        5.0,
        None,
        None,
        0,
        None,
        None,
        0.1,
        None,
        None,
        None,
        None,
        analyses,
    )
    assert rec.analyses is analyses
    assert rec.checkpoint_tableaux is None
    assert rec.checkpoint_values is None
