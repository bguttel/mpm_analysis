"""
Synthetic braids with analytically known invariants.

Each case is built so the answer is exact rather than fitted: two eigenvalues
counter-rotating as ``±½ r e^{i n k}`` have a gap ``r e^{i n k}`` whose winding
over one Brillouin zone is exactly ``n``.  Choosing ``n`` half-integer makes the
pair swap after one loop — a 4π component — and a rotating triangle makes a 6π
one.  That covers precisely the cases the single-loop analysis cannot represent.

Reference: Pavlov, Gefen, Shnirman, Phys. Rev. B 111, 104301 (2025), Table I.
"""
from __future__ import annotations

import numpy as np
import pytest

from mpm_analysis.analysis.braid_topology import (
    analyze_braid,
    canonical_word,
    close_loop,
    exponent_sum,
    nu_group,
    nu_ordered,
    nu_pair,
    unroll,
    winding_curve,
)

# Open k-grid, deliberately missing k=2π — the same shape the experiment
# produces once K_INDEX_RANGE trims the trivial endpoints.
K = np.linspace(0.0, 2.0 * np.pi, 721)[:-1]


def rotating_pair(n: float, third: complex = 3.0 + 0j) -> np.ndarray:
    """Two strands winding ``n`` times about each other, plus a distant third."""
    z = np.zeros((len(K), 3), dtype=complex)
    z[:, 0] = 0.5 * np.exp(1j * n * K)
    z[:, 1] = -0.5 * np.exp(1j * n * K)
    z[:, 2] = third
    return z


def rotating_triangle() -> np.ndarray:
    """Three strands on a triangle advancing one slot per loop -> 6π."""
    return np.stack(
        [np.exp(1j * (2 * np.pi * j / 3 + K / 3)) for j in range(3)], axis=1
    )


# ---------------------------------------------------------------------------
# Paper Table I classes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n, expect_class, expect_nu, expect_period, expect_laps",
    [
        (0.0, "V", 0.0, "2π + 2π + 2π", 1),
        (1.0, "II", 2.0, "2π + 2π + 2π", 1),
        (0.5, "III", 1.0, "4π + 2π", 2),
    ],
)
def test_table_i_classes(n, expect_class, expect_nu, expect_period, expect_laps):
    topo = analyze_braid(rotating_pair(n), K)
    assert topo.paper_class == expect_class
    assert topo.nu_total == pytest.approx(expect_nu, abs=1e-6)
    assert topo.period_label == expect_period
    assert topo.unrolled.n_laps == expect_laps
    assert topo.quality.trustworthy


def test_six_pi_component_is_class_iv():
    topo = analyze_braid(rotating_triangle(), K)
    assert topo.paper_class == "IV"
    assert topo.nu_total == pytest.approx(2.0, abs=1e-6)
    assert topo.period_label == "6π"
    assert topo.unrolled.n_laps == 3
    # a single 3-strand component, i.e. one closed loop of the link
    assert len(topo.unrolled.components) == 1
    assert sorted(topo.unrolled.components[0]) == [0, 1, 2]


def test_mirror_class_is_recognised_and_labelled():
    topo = analyze_braid(rotating_pair(-1.0), K)
    assert topo.paper_class == "II (mirror)"
    assert topo.nu_total == pytest.approx(-2.0, abs=1e-6)


def test_double_braiding_is_not_forced_into_a_paper_class():
    """σ2⁴ closes to a (2,4) torus link, which is not in Table I."""
    topo = analyze_braid(rotating_pair(2.0), K)
    assert topo.nu_total == pytest.approx(4.0, abs=1e-6)
    assert topo.paper_class is None
    assert "unclassified" in topo.closure_label


# ---------------------------------------------------------------------------
# Loop closure — the thing that makes the indices quantized
# ---------------------------------------------------------------------------

def test_closure_must_apply_the_permutation():
    """Closing a 4π pair without the permutation destroys the half turn."""
    z = rotating_pair(0.5)
    swap = np.array([1, 0, 2])

    wrong, _, _ = close_loop(z, K, np.arange(3))     # naive identity closure
    right, _, _ = close_loop(z, K, swap)             # correct permuted closure

    assert nu_ordered(wrong, 0, 1) == pytest.approx(0.0, abs=1e-3)
    assert nu_ordered(right, 0, 1) == pytest.approx(0.5, abs=1e-3)


def test_indices_are_quantized_after_closure():
    for n in (0.0, 0.5, 1.0, 1.5, 2.0):
        topo = analyze_braid(rotating_pair(n), K)
        assert topo.quality.integer_defect < 1e-6, f"n={n}"


def test_closure_defect_is_small_on_a_fine_grid():
    topo = analyze_braid(rotating_pair(1.0), K)
    assert topo.quality.closure_defect < 0.05


# ---------------------------------------------------------------------------
# Unrolling
# ---------------------------------------------------------------------------

def test_unrolled_component_closes_only_after_its_own_period():
    """A 4π pair reaches an integer winding on lap 2, not lap 1."""
    z = rotating_pair(0.5)
    perm = np.array([1, 0, 2])
    closed, k_closed, _ = close_loop(z, K, perm)
    un = unroll(closed, k_closed, perm)

    assert un.n_laps == 2
    cum, total = winding_curve(un.strands[:, 0] - un.strands[:, 1])
    assert total == pytest.approx(1.0, abs=1e-3)

    half = int(np.argmin(np.abs(un.k - (un.k[0] + 2 * np.pi))))
    assert cum[half] == pytest.approx(0.5, abs=1e-2)      # mid-period: not integer


def test_unrolled_strands_return_to_their_start():
    z = rotating_triangle()
    perm = np.array([1, 2, 0])
    closed, k_closed, _ = close_loop(z, K, perm)
    un = unroll(closed, k_closed, perm)
    assert un.n_laps == 3
    assert np.allclose(un.strands[0], un.strands[-1], atol=1e-8)


# ---------------------------------------------------------------------------
# Index algebra (paper Eqs. 9-13)
# ---------------------------------------------------------------------------

def test_nu_pair_is_twice_nu_ordered():
    """Δ_ba = −Δ_ab has the same winding, so ν_{a;b} = 2 ν^(a,b) exactly."""
    z = rotating_pair(1.0)
    closed, _, _ = close_loop(z, K, np.arange(3))
    assert nu_pair(closed, 0, 1) == pytest.approx(2 * nu_ordered(closed, 0, 1))
    assert nu_ordered(closed, 0, 1) == pytest.approx(nu_ordered(closed, 1, 0))


def test_nu_group_sums_pairs():
    z = rotating_pair(1.0)
    closed, _, _ = close_loop(z, K, np.arange(3))
    expected = nu_pair(closed, 0, 2) + nu_pair(closed, 1, 2)
    assert nu_group(closed, [0, 1], [2]) == pytest.approx(expected)


def test_nu_group_rejects_overlapping_groups():
    z = rotating_pair(1.0)
    with pytest.raises(ValueError):
        nu_group(z, [0, 1], [1, 2])


def test_nu_matches_upstream_invariant_sum():
    """Cross-check against the validated Theory/braid_tools_complete.py."""
    bt = pytest.importorskip("mpm_analysis.analysis.braiding").load_braid_tools()
    for n in (0.0, 0.5, 1.0, 2.0):
        z = rotating_pair(n)
        perm = bt.closing_permutation(z)
        closed, _, _ = close_loop(z, K, perm)
        mine = analyze_braid(z, K).nu_total
        assert mine == pytest.approx(bt.invariant_sum(closed, range(3)), abs=1e-6)


def test_nu_equals_braid_word_exponent_sum():
    """ν (Eq. 13) is the writhe of the braid word — a conjugacy invariant."""
    for n in (0.0, 0.5, 1.0, 2.0):
        topo = analyze_braid(rotating_pair(n), K)
        assert exponent_sum(topo.word) == pytest.approx(topo.nu_total, abs=1e-6)


# ---------------------------------------------------------------------------
# Canonical braid word
# ---------------------------------------------------------------------------

def test_braid_relation_is_absorbed():
    """σ1σ2σ1 = σ2σ1σ2 must reach the same representative."""
    assert canonical_word([(1, 1), (2, 1), (1, 1)]) == \
           canonical_word([(2, 1), (1, 1), (2, 1)])


def test_canonical_word_is_cyclic_invariant():
    base = [(1, 1), (2, 1), (2, 1)]
    for r in range(len(base)):
        rotated = base[r:] + base[:r]
        assert canonical_word(rotated) == canonical_word(base)


def test_adjacent_inverses_cancel():
    assert canonical_word([(1, 1), (2, 1), (2, -1), (1, -1)]) == []
    assert canonical_word([]) == []


def test_exponent_sum_is_preserved_by_canonicalization():
    for word in ([(1, 1), (2, 1), (1, 1)], [(2, 1), (2, 1)], [(1, -1), (2, 1)]):
        assert exponent_sum(canonical_word(word)) == exponent_sum(word)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

def test_nan_strands_are_dropped_not_renumbered():
    z = rotating_pair(1.0)
    z[:, 2] = np.nan
    topo = analyze_braid(z, K)
    assert topo.quality.n_strands_dropped == 1
    assert topo.strand_labels == [0, 1]          # original labels preserved
    assert topo.nu_total == pytest.approx(2.0, abs=1e-6)


def test_single_strand_is_not_a_braid():
    z = np.ones((len(K), 1), dtype=complex)
    topo = analyze_braid(z, K)
    assert not topo.quality.trustworthy
    assert topo.closure_label == "no braid"


def test_winding_survives_a_near_origin_pass():
    """A tiny gap must degrade the quality flag, not NaN out the index."""
    z = rotating_pair(1.0)
    z[:, 0] *= 1e-9                                # strands 0,1 nearly collide
    z[:, 1] *= 1e-9
    topo = analyze_braid(z, K)
    assert np.isfinite(topo.nu_total)
    assert topo.quality.min_gap < 1e-6


# ---------------------------------------------------------------------------
# Panels build without error on every case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maker", [
    lambda: rotating_pair(0.0),
    lambda: rotating_pair(1.0),
    lambda: rotating_pair(0.5),
    rotating_triangle,
])
def test_panel_geometry_builders(maker):
    from mpm_analysis.plotting.braid_panels import (
        braid_diagram_geometry,
        link_graph_geometry,
        winding_curves_geometry,
    )

    topo = analyze_braid(maker(), K)

    gA = braid_diagram_geometry(topo)
    assert gA.height.shape[0] == len(gA.k)
    assert len(gA.seams) == topo.unrolled.n_laps - 1
    assert len(gA.crossing_label) == len(gA.crossing_k)

    gB = winding_curves_geometry(topo)
    assert all(len(c) == len(gB.k) for c in gB.curves)

    gC = link_graph_geometry(topo)
    assert len(gC.node_labels) == len(topo.unrolled.components)
