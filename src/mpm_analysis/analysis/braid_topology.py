"""
braid_topology.py
-----------------
Braid invariants of non-Hermitian eigenvalue strands, following

    A. I. Pavlov, Y. Gefen, A. Shnirman,
    "Topological transitions in quantum jump dynamics: Hidden exceptional points",
    Phys. Rev. B 111, 104301 (2025).

This module is self-contained: the closing permutation and cycle
decomposition live here (moved from the historical
``Theory/braid_tools_complete.py``), alongside everything the braid diagrams
and the paper's classification actually need:

1. **Loop closure.**  The experimental k-grid is open — ``K_INDEX_RANGE=(1,-1)``
   trims k=0 and k=2π because ``P(k=0,t) ≡ 1`` is trivially constant — so the
   paper's contour integrals run over an open path and ν comes out a few percent
   off an integer.  Since ``L_k = L_{k+2π}``, the first sample *is* the sample
   one grid step past the last, so appending it closes the loop exactly up to
   grid resolution.  See :func:`close_loop`.

2. **Unrolling.**  A component whose closing permutation cycle has length ``L``
   is ``2πL``-periodic: it only returns to itself after ``L`` traversals of the
   Brillouin zone.  Upstream detects this (``cycles_of`` / ``period_label``) but
   never tiles the loop, so a 4π or 6π component can be *counted* and never
   *seen*, and its pair winding is not individually quantized.
   :func:`unroll` builds the multi-lap strands on which each component genuinely
   closes; this is what makes the braid diagram and the winding accumulator
   meaningful for the interesting cases.

3. **A canonical braid word.**  Upstream ``reduce_braid_word`` cancels only
   adjacent inverses.  The paper's classes are defined up to cyclic permutation
   (§IV and Appendix B), so that is not enough to recognise a class.
   :func:`canonical_word` additionally applies the braid relation and minimises
   over cyclic rotations.

4. **An honest classification.**  Upstream ``classify_three_strand_closure`` is
   hardcoded to three strands, ignores the braid word, and emits ``"Class-II-like:
   …"`` guesses.  :func:`classify` matches a full invariant fingerprint against
   Table I of the paper and reports the fingerprint verbatim when nothing
   matches, instead of guessing.

Index conventions
~~~~~~~~~~~~~~~~~
The paper's ordered index (Eq. 9) is the winding of the *gap*::

    ν^(a,b) = (1/2πi) ∮ dk  d/dk  log[ (λ_a(k) − λ_b(k)) / 2 ]

and ν_{a;b} = ν^(a,b) + ν^(b,a) (Eq. 10).  Because Δ_ba = −Δ_ab has the *same*
winding as Δ_ab, the two ordered indices are equal and therefore

    ν_{a;b} = 2 · ν^(a,b)        exactly.

So the natural "one full counterclockwise loop = +1" normalization used by
:func:`winding_curve` reads out **ν^(a,b)**, and the paper's pair index is twice
the endpoint height.  The linking number of the two closed curves is
lk(a,b) = ν_{a;b}/2 = ν^(a,b).  Angles are counterclockwise-positive in the
ordinary (Re, Im) plane.

ν (Eq. 13) is the total braid degree = the exponent sum (writhe) of the braid
word — a genuine conjugacy invariant, and the most trustworthy single number
here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import gcd
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Loop-permutation helpers (moved here from Theory/braid_tools_complete.py so
# this module is self-contained; previously reached via load_braid_tools).
# ---------------------------------------------------------------------------


def closing_permutation(eigs_loop: np.ndarray) -> np.ndarray:
    """perm[i] = strand index at k=0 reached by strand i after one 2π loop."""
    cost = np.abs(eigs_loop[-1][:, None] - eigs_loop[0][None, :])
    row, col = linear_sum_assignment(cost)
    perm = np.empty(eigs_loop.shape[1], dtype=int)
    perm[row] = col
    return perm


def cycles_of(perm: Sequence[int]) -> list[list[int]]:
    """Cycle decomposition of a permutation."""
    n = len(perm)
    seen = [False] * n
    cycles: list[list[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        cyc = []
        j = i
        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = int(perm[j])
        cycles.append(cyc)
    return cycles


def period_label(cycles: Sequence[Sequence[int]]) -> str:
    """``"4π + 2π"``-style label from cycle lengths."""
    lengths = sorted((len(c) for c in cycles), reverse=True)
    return " + ".join(f"{2 * L}π" for L in lengths)

# Projection used to turn complex strands into a planar braid diagram.  Matches
# ``braid_tools_complete.braid_word_from_eigs`` so crossing signs agree with the
# upstream word: the height is p = Re + mix·Im and over/under is decided by
# q = Im − mix·Re.  The irrational mix breaks accidental exact ties.
DEFAULT_PROJ_MIX = float(np.sqrt(2.0) / 10.0)


# ---------------------------------------------------------------------------
# Loop closure and unrolling
# ---------------------------------------------------------------------------

def close_loop(
    strands: np.ndarray,
    k: np.ndarray,
    perm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Append the periodic wrap sample so the k-path is genuinely closed.

    ``L_k`` is 2π-periodic in the counting field, so the strand values at the
    first sampled k *are* the values one grid step past the last.  Appending them
    (at ``k_last + Δk``) turns the open sampled path into a closed contour, which
    is what quantizes the winding integrals.  Nothing is interpolated and no
    endpoint is invented — the trimmed k=0 / k=2π samples stay excluded.

    The wrap sample must be **permuted**: a strand that is not 2π-periodic does
    not come back to its own column after one Brillouin zone, it arrives at
    ``perm[j]``.  Appending ``strands[0]`` unpermuted would silently snap such a
    strand back to its starting value and destroy exactly the half-turn that
    makes it a 4π component — a 4π pair would read ν = 0 instead of ν = 1.

    Parameters
    ----------
    strands : (n_k, M) complex
    k : (n_k,) float
    perm : (M,) int
        Closing permutation from the *open* path
        (``braid_tools_complete.closing_permutation``): ``perm[j]`` is the strand
        that strand ``j`` continues into after one loop.

    Returns
    -------
    (strands_closed, k_closed, closure_defect)
        ``closure_defect`` is the largest ``|z_last − z_wrap|`` over strands, in
        the same units as the poles.  A large value means the k-grid is too
        coarse (or the strands are mis-sorted) and the indices are unreliable.
    """
    strands = np.asarray(strands, dtype=complex)
    k = np.asarray(k, dtype=float)
    perm = np.asarray(perm, dtype=int)
    if strands.ndim != 2:
        raise ValueError(f"strands must be 2-D (n_k, M); got shape {strands.shape}")
    if len(k) != strands.shape[0]:
        raise ValueError(f"k has {len(k)} samples but strands has {strands.shape[0]}")
    if len(k) < 2:
        return strands.copy(), k.copy(), float("nan")

    dk = float(k[-1] - k[-2])
    wrap = strands[0][perm]
    defect = float(np.nanmax(np.abs(wrap - strands[-1])))
    return (
        np.vstack([strands, wrap[None, :]]),
        np.concatenate([k, [k[-1] + dk]]),
        defect,
    )


@dataclass
class UnrolledBraid:
    """Strands tiled over as many Brillouin zones as it takes them to close.

    Attributes
    ----------
    k : (n_ext,) float
        Extended k axis, spanning ``[k0, k0 + n_laps·2π]``.
    strands : (n_ext, M) complex
        Column ``s`` follows one physical strand across the seams.
    n_laps : int
        Number of 2π traversals drawn = lcm of the cycle lengths.
    component_of_strand : (M,) int
        Which closing-permutation cycle each column belongs to.
    components : list[list[int]]
        The cycles themselves (strand indices).
    period_laps : list[int]
        Cycle length per component: 1 → 2π-periodic, 2 → 4π, 3 → 6π.
    seams : (n_laps - 1,) float
        k values where one Brillouin zone ends and the next begins.
    """

    k: np.ndarray
    strands: np.ndarray
    n_laps: int
    component_of_strand: np.ndarray
    components: list[list[int]]
    period_laps: list[int]
    seams: np.ndarray

    @property
    def n_strands(self) -> int:
        return self.strands.shape[1]

    def component_strands(self, comp_idx: int) -> np.ndarray:
        """The (n_ext, len(cycle)) block of strands forming one component."""
        return self.strands[:, self.components[comp_idx]]


def _perm_power(perm: np.ndarray, power: int) -> np.ndarray:
    """``perm`` composed with itself ``power`` times (identity for power=0)."""
    out = np.arange(len(perm), dtype=int)
    for _ in range(power):
        out = perm[out]
    return out


def _lcm(values: Iterable[int]) -> int:
    out = 1
    for v in values:
        v = max(1, int(v))
        out = out * v // gcd(out, v)
    return out


def unroll(
    strands: np.ndarray,
    k: np.ndarray,
    perm: np.ndarray,
    *,
    n_laps: int | None = None,
) -> UnrolledBraid:
    """Tile the closed k-loop until every component returns to itself.

    Lap ``ℓ`` of the drawing shows the same 2π of physics, but read out through
    ``perm**ℓ`` — that is what following a strand across the Brillouin-zone seam
    means.  After ``lcm(cycle lengths)`` laps every strand is back where it
    started, so the drawn curves are closed and their pairwise windings are
    honest integers.

    A 2-cycle therefore unrolls to two laps (a 4π component) and a 3-cycle to
    three (6π) — the cases the single-loop analysis cannot represent.

    Parameters
    ----------
    strands : (n_k, M) complex
        Should already be loop-closed (see :func:`close_loop`).
    k : (n_k,) float
    perm : (M,) int
        Closing permutation, ``perm[i]`` = strand reached by strand ``i`` after
        one loop (``braid_tools_complete.closing_permutation``).
    n_laps :
        Override the number of laps.  Default: the lcm of the cycle lengths.
    """
    strands = np.asarray(strands, dtype=complex)
    k = np.asarray(k, dtype=float)
    perm = np.asarray(perm, dtype=int)
    M = strands.shape[1]

    components = [list(c) for c in cycles_of(perm)]
    period_laps = [len(c) for c in components]
    if n_laps is None:
        n_laps = _lcm(period_laps) if period_laps else 1

    comp_of = np.zeros(M, dtype=int)
    for ci, cyc in enumerate(components):
        for s in cyc:
            comp_of[s] = ci

    span = float(k[-1] - k[0])
    # Drop the duplicated seam sample on every lap after the first, so the
    # concatenated curve has no repeated point.
    k_parts = [k]
    s_parts = [strands]
    for lap in range(1, n_laps):
        k_parts.append(k[1:] + lap * span)
        s_parts.append(strands[1:][:, _perm_power(perm, lap)])

    seams = np.array([k[0] + lap * span for lap in range(1, n_laps)], dtype=float)
    return UnrolledBraid(
        k=np.concatenate(k_parts),
        strands=np.vstack(s_parts),
        n_laps=int(n_laps),
        component_of_strand=comp_of,
        components=components,
        period_laps=period_laps,
        seams=seams,
    )


# ---------------------------------------------------------------------------
# Winding indices (paper Eqs. 9-13)
# ---------------------------------------------------------------------------

def winding_curve(z: np.ndarray) -> tuple[np.ndarray, float]:
    """Cumulative and total winding of a sampled complex curve, in loops.

    ``cum[i]`` is the turning accumulated from the first sample to sample ``i``,
    normalized so one full counterclockwise loop about the origin is ``+1``;
    ``total = cum[-1]``.  Steps are summed as principal-value increments
    ``arg(z[i+1]/z[i])``, the same estimator as
    ``braid_tools_complete.phase_winding``.

    Unlike that function this one does **not** collapse to NaN when the curve
    passes close to the origin: a near-exceptional-point sample is reported
    through :class:`BraidQuality` instead of destroying the index.  Exact zeros
    are the one unrecoverable case and do yield NaN.
    """
    z = np.asarray(z, dtype=complex)
    if z.size < 2:
        return np.zeros(z.size), float("nan")
    if np.any(z == 0):
        return np.full(z.size, np.nan), float("nan")
    steps = np.angle(z[1:] / z[:-1])
    cum = np.concatenate(([0.0], np.cumsum(steps))) / (2.0 * np.pi)
    return cum, float(cum[-1])


def nu_ordered(strands: np.ndarray, a: int, b: int) -> float:
    """ν^(a,b) of paper Eq. 9 — the winding of the gap (λ_a − λ_b)/2."""
    _, total = winding_curve(
        (np.asarray(strands, dtype=complex)[:, a] - np.asarray(strands, dtype=complex)[:, b]) / 2.0
    )
    return total


def nu_pair(strands: np.ndarray, a: int, b: int) -> float:
    """ν_{a;b} of paper Eq. 10.  Equals ``2·nu_ordered`` (see module docstring)."""
    return 2.0 * nu_ordered(strands, a, b)


def nu_group(strands: np.ndarray, group_a: Sequence[int], group_b: Sequence[int]) -> float:
    """ν_{A;B} of paper Eqs. 11-12 for arbitrary strand groupings.

    ``ν_{a,b;c} = ν^(a,c) + ν^(c,a) + ν^(b,c) + ν^(c,b)`` generalises to the sum
    of ν_{i;j} over ``i ∈ A``, ``j ∈ B``.  Upstream only exposes this for the
    permutation cycles (``link_sum``); here any grouping is allowed, which is
    what Eqs. 11 and 12 actually define.
    """
    a_set, b_set = list(group_a), list(group_b)
    if set(a_set) & set(b_set):
        raise ValueError("nu_group needs disjoint groups")
    return float(sum(nu_pair(strands, i, j) for i in a_set for j in b_set))


def self_writhe(strands: np.ndarray, group: Sequence[int]) -> float:
    """Self-braiding of one component: Σ_{i<j∈A} ν_{i;j} (upstream ``invariant_sum``)."""
    members = list(group)
    if len(members) < 2:
        return 0.0
    return float(sum(nu_pair(strands, i, j) for i, j in combinations(members, 2)))


def nu_total(strands: np.ndarray) -> float:
    """ν of paper Eq. 13 — the total braid degree over all strands."""
    return self_writhe(strands, range(np.asarray(strands).shape[1]))


def pair_matrix(strands: np.ndarray) -> np.ndarray:
    """Symmetric (M, M) matrix of ν^(a,b); the diagonal is zero."""
    strands = np.asarray(strands, dtype=complex)
    M = strands.shape[1]
    out = np.zeros((M, M), dtype=float)
    for i, j in combinations(range(M), 2):
        out[i, j] = out[j, i] = nu_ordered(strands, i, j)
    return out


def linking_matrix(strands: np.ndarray, components: Sequence[Sequence[int]]) -> np.ndarray:
    """(C, C) matrix of linking numbers between closed components.

    Off-diagonal: ``lk(A,B) = ν_{A;B}/2``, the ordinary linking number of the two
    closed curves.  Diagonal: the component's self-writhe divided by 2, i.e. how
    much the strands of a multi-strand (4π/6π) component wind about each other.
    """
    comps = [list(c) for c in components]
    n = len(comps)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        out[i, i] = self_writhe(strands, comps[i]) / 2.0
    for i, j in combinations(range(n), 2):
        out[i, j] = out[j, i] = nu_group(strands, comps[i], comps[j]) / 2.0
    return out


# ---------------------------------------------------------------------------
# Braid word
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Crossing:
    """One generator applied at a definite place along the k axis."""

    k: float
    generator: int      # 1-based, as in σ_1, σ_2, …
    sign: int           # +1 for σ_i, -1 for σ_i^{-1}
    strand_over: int
    strand_under: int


def crossings(
    strands: np.ndarray,
    k: np.ndarray,
    *,
    proj_mix: float = DEFAULT_PROJ_MIX,
) -> list[Crossing]:
    """Locate every strand exchange along the k axis.

    Same planar projection as ``braid_tools_complete.braid_word_from_eigs`` — the
    height is ``p = Re + mix·Im``, and the strand with the larger
    ``q = Im − mix·Re`` passes over — so signs agree with the upstream word.  The
    difference is that this keeps the k position of each crossing, which upstream
    discards and which the braid diagram needs.

    Call it on the **closed single loop** to get the braid word whose closure is
    the link (that is what paper Table I tabulates), and on the **unrolled**
    strands to draw the multi-lap diagram.

    The crossing k is refined to sub-grid resolution by linear interpolation of
    the height difference through its zero, so markers land on the crossing
    rather than on the nearest sample.
    """
    z = np.asarray(strands, dtype=complex)
    k = np.asarray(k, dtype=float)
    M = z.shape[1]
    if M < 2:
        return []

    p = z.real + proj_mix * z.imag
    q = z.imag - proj_mix * z.real

    out: list[Crossing] = []
    order = list(np.argsort(-p[0], kind="mergesort"))

    for n in range(len(p) - 1):
        target = list(np.argsort(-p[n + 1], kind="mergesort"))
        pos_in_target = {s: i for i, s in enumerate(target)}
        guard = 0
        while order != target and guard < M * M:
            guard += 1
            swapped = False
            for pos in range(M - 1):
                a, b = order[pos], order[pos + 1]
                if pos_in_target[a] > pos_in_target[b]:
                    # sub-grid crossing position from the height difference
                    d0 = p[n, a] - p[n, b]
                    d1 = p[n + 1, a] - p[n + 1, b]
                    frac = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.5
                    frac = float(np.clip(frac, 0.0, 1.0))
                    k_cross = float(k[n] + frac * (k[n + 1] - k[n]))

                    qa = 0.5 * (q[n, a] + q[n + 1, a])
                    qb = 0.5 * (q[n, b] + q[n + 1, b])
                    sign = 1 if qa > qb else -1
                    over, under = (a, b) if qa > qb else (b, a)
                    out.append(Crossing(k_cross, pos + 1, sign, int(over), int(under)))

                    order[pos], order[pos + 1] = order[pos + 1], order[pos]
                    swapped = True
            if not swapped:
                break
        order = target

    return out


def word_from_crossings(cross: Sequence[Crossing]) -> list[tuple[int, int]]:
    """``[(generator, sign), …]`` in k order."""
    return [(c.generator, c.sign) for c in sorted(cross, key=lambda c: c.k)]


def _free_reduce(word: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Cancel adjacent inverses, and commute far generators to expose more."""
    out: list[tuple[int, int]] = []
    for gen, sign in word:
        cancelled = False
        # walk back over generators that commute with this one (|i-j| >= 2)
        for back in range(len(out) - 1, -1, -1):
            g2, s2 = out[back]
            if g2 == gen:
                if s2 == -sign:
                    out.pop(back)
                    cancelled = True
                break
            if abs(g2 - gen) < 2:
                break                      # blocked: does not commute
        if not cancelled:
            out.append((int(gen), int(sign)))
    return out


def _braid_relation_variants(word: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """All words reachable by one application of σiσjσi = σjσiσj (|i-j| = 1)."""
    w = list(word)
    out: list[list[tuple[int, int]]] = []
    for i in range(len(w) - 2):
        (g0, s0), (g1, s1), (g2, s2) = w[i], w[i + 1], w[i + 2]
        if abs(g0 - g1) == 1 and g0 == g2 and s0 == s1 == s2:
            out.append(w[:i] + [(g1, s1), (g0, s0), (g1, s1)] + w[i + 3:])
    return out


def canonical_word(
    word: Sequence[tuple[int, int]],
    *,
    max_search: int = 4096,
) -> list[tuple[int, int]]:
    """A canonical representative of the braid's conjugacy class.

    The closed braid — and therefore the topological class — is invariant under
    cyclic permutation of the word (paper §IV: "braid words are defined up to
    cyclic permutations"), under free reduction, under commutation of far
    generators, and under the braid relation σiσjσi = σjσiσj.  This explores that
    equivalence by breadth-first search and returns the smallest word found,
    ordered by (length, tuple).

    Honest limitation: this is a practical canonical form, not a proven complete
    conjugacy invariant for B_n.  Two words that reduce to the same
    representative are certainly equivalent; two that do not *may* still be.
    The parts of the fingerprint that are provably invariant are the exponent sum
    (:func:`exponent_sum`, = ν of Eq. 13), the cycle type of the closing
    permutation, and the linking matrix — classification leans on those first.
    """
    start = tuple(_free_reduce(word))
    best = start
    seen = {start}
    queue = [start]

    while queue and len(seen) < max_search:
        cur = queue.pop(0)
        cur_l = list(cur)
        neighbours = [cur_l[r:] + cur_l[:r] for r in range(len(cur_l))]  # cyclic
        neighbours += _braid_relation_variants(cur_l)
        for cand in neighbours:
            red = tuple(_free_reduce(cand))
            if red in seen:
                continue
            seen.add(red)
            queue.append(red)
            if (len(red), red) < (len(best), best):
                best = red

    return [tuple(g) for g in best]


def exponent_sum(word: Sequence[tuple[int, int]]) -> int:
    """Writhe of the braid word — a conjugacy invariant, equal to ν (Eq. 13)."""
    return int(sum(sign for _, sign in word))


def word_to_string(word: Sequence[tuple[int, int]]) -> str:
    """``σ2 σ2`` / ``σ1 σ2⁻¹`` / ``1`` for the identity."""
    if not len(word):
        return "1"
    return " ".join(f"σ{g}" + ("" if s > 0 else "⁻¹") for g, s in word)


# ---------------------------------------------------------------------------
# Classification against the paper's Table I
# ---------------------------------------------------------------------------

#: Paper Table I (p. 5).  Classes are matched on the *quantized* part of the
#: fingerprint — cycle type of the closing permutation, ν (Eq. 13), and the
#: multiset of pair indices where Table I pins them.  Those three already
#: separate all five classes uniquely, and unlike the braid word they are
#: invariant under the strand-labelling convention.  ``word`` is kept for
#: reference and cross-checked against ν, but is deliberately not a match
#: criterion: braid words are defined only up to conjugacy (paper §IV), and
#: conjugation by the half twist swaps σ1 ↔ σ2, so a perfectly good Class II
#: measurement can print σ1σ1 instead of σ2σ2.
_PAPER_CLASSES: list[dict] = [
    {
        "name": "I",
        "cycle_type": (1, 1, 1),
        "nu": 4,
        "pairs": (0, 2, 2),
        "word": [(1, 1), (2, 1), (2, 1), (1, 1)],       # σ1σ2σ2σ1
        "closure": "connected sum of two Hopf links",
    },
    {
        "name": "II",
        "cycle_type": (1, 1, 1),
        "nu": 2,
        "pairs": (0, 0, 2),
        "word": [(2, 1), (2, 1)],                        # σ2σ2
        "closure": "Hopf link + unlink",
    },
    {
        "name": "III",
        "cycle_type": (2, 1),
        "nu": 1,
        "pairs": None,
        "word": [(2, 1)],                                # σ2
        "closure": "unknot + unlink (4π component)",
    },
    {
        "name": "IV",
        "cycle_type": (3,),
        "nu": 2,
        "pairs": None,
        "word": [(2, 1), (1, 1)],                        # σ2σ1
        "closure": "three-component unknot (6π)",
    },
    {
        "name": "V",
        "cycle_type": (1, 1, 1),
        "nu": 0,
        "pairs": (0, 0, 0),
        "word": [],                                      # identity
        "closure": "three unlinks",
    },
]



@dataclass
class BraidQuality:
    """How much the numbers below should be trusted."""

    min_gap: float = float("nan")
    min_gap_k: float = float("nan")
    integer_defect: float = float("nan")
    closure_defect: float = float("nan")
    n_strands_dropped: int = 0
    trustworthy: bool = True
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"min|Δ|={self.min_gap:.3g} at k={self.min_gap_k:.3f}",
                f"int. defect={self.integer_defect:.3f}"]
        if np.isfinite(self.closure_defect):
            bits.append(f"closure={self.closure_defect:.3g}")
        if self.n_strands_dropped:
            bits.append(f"{self.n_strands_dropped} strand(s) dropped")
        return "  |  ".join(bits)


@dataclass
class BraidTopology:
    """Everything known about the braid at one λ."""

    lambda_val: float
    unrolled: UnrolledBraid
    crossings: list[Crossing]           # over one loop; their closure is the link
    unrolled_crossings: list[Crossing]  # over the full period; for the diagram
    word: list[tuple[int, int]]
    word_str: str
    nu_total: float
    pair_nu: dict[str, float]          # ν_{a;b}, paper convention (= 2·winding)
    ordered_nu: dict[str, float]       # ν^(a,b), one loop = 1
    pair_windings: np.ndarray          # (M, M) of ν^(a,b)
    linking: np.ndarray                # (C, C) between components
    period_label: str
    cycle_type: tuple[int, ...]
    paper_class: str | None
    closure_label: str
    quality: BraidQuality
    strand_labels: list[int]           # original pole-track index per column

    @property
    def n_components(self) -> int:
        return len(self.unrolled.components)

    @property
    def class_label(self) -> str:
        """Short human label for panels: paper class if matched, else fingerprint."""
        if self.paper_class is not None:
            return f"Class {self.paper_class}"
        return f"ν={self.nu_total:+.0f}, {self.period_label}"


def classify(
    cycle_type: tuple[int, ...],
    nu: float,
    pair_nu: dict[str, float],
    word: Sequence[tuple[int, int]],
    *,
    tol: float = 5e-2,
) -> tuple[str | None, str]:
    """Match an invariant fingerprint against paper Table I.

    Returns ``(class_name_or_None, closure_label)``.  A class is claimed only
    when the cycle type, ν, the canonical word and (where Table I pins them) the
    pair indices *all* agree.  When nothing matches, the fingerprint itself is
    returned as the label — never a ``"-like"`` guess.

    ``word`` must already be canonical (:func:`canonical_word`).
    """
    if not np.isfinite(nu):
        return None, "indices not finite (sampled exceptional point?)"

    def _match(nu_val: float, pair_vals: tuple[int, ...]) -> dict | None:
        for entry in _PAPER_CLASSES:
            if tuple(entry["cycle_type"]) != cycle_type:
                continue
            if abs(nu_val - entry["nu"]) > tol:
                continue
            if entry["pairs"] is not None and pair_vals != tuple(entry["pairs"]):
                continue
            return entry
        return None

    pairs = tuple(sorted(round(v) for v in pair_nu.values()))
    entry = _match(nu, pairs)
    suffix = ""
    if entry is None:
        # A globally sign-flipped braid is the mirror link: the same class
        # traversed with the opposite orientation.  Worth naming rather than
        # dropping, since the overall sign here is fixed by the strand-sorting
        # convention, not by physics.
        entry = _match(-nu, tuple(sorted(-v for v in pairs)))
        suffix = " (mirror)"

    if entry is not None:
        name = entry["name"] + suffix
        closure = entry["closure"] + suffix
        # The word is not part of the match — σ1σ1 and σ2σ2 are conjugate in B_3
        # (conjugation by the half twist swaps the generators), so requiring the
        # printed word to agree would reject correct classifications.  Instead
        # the word is cross-checked for consistency and reported.
        if word and abs(abs(nu) - abs(exponent_sum(word))) > 0.5:
            closure += f"  [word {word_to_string(word)} disagrees with ν]"
        return name, closure

    period = " + ".join(f"{2 * c}π" for c in cycle_type)
    return None, (f"unclassified: ν={nu:+.3g}, periods {period}, "
                  f"word {word_to_string(word)}")


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def analyze_braid(
    strands: np.ndarray,
    k: np.ndarray,
    *,
    lambda_val: float = float("nan"),
    strand_labels: Sequence[int] | None = None,
    integer_tol: float = 5e-2,
    proj_mix: float = DEFAULT_PROJ_MIX,
) -> BraidTopology:
    """Full braid characterization of one λ slice.

    Pipeline: drop all-NaN strands → close the loop → closing permutation →
    unroll to the common period → indices, crossings and canonical word on the
    unrolled braid → classify.

    Parameters
    ----------
    strands : (n_k, M) complex
        Continuity-sorted eigenvalue strands over one Brillouin zone.  Use
        ``mpm_analysis.kspace.plotting.track_permutation`` upstream so the strand
        index means the same branch at every λ.
    k : (n_k,) float
    lambda_val :
        Recorded for labelling only.
    strand_labels :
        Original pole-track indices, kept through NaN dropping so generators are
        never silently renumbered in the report.
    integer_tol :
        How far ν may sit from an integer before the result is flagged untrusted.
    """
    strands = np.asarray(strands, dtype=complex)
    k = np.asarray(k, dtype=float)
    M0 = strands.shape[1]
    labels = list(range(M0)) if strand_labels is None else [int(s) for s in strand_labels]

    quality = BraidQuality()

    # --- drop strands that are absent at this λ (pole_order < max_order) ------
    keep = [j for j in range(M0) if np.all(np.isfinite(strands[:, j]))]
    quality.n_strands_dropped = M0 - len(keep)
    if quality.n_strands_dropped:
        quality.messages.append(
            f"{quality.n_strands_dropped} of {M0} strands are not finite over the "
            f"whole loop and were dropped; generators refer to the remaining "
            f"{len(keep)}."
        )
    strands = strands[:, keep]
    labels = [labels[j] for j in keep]
    M = strands.shape[1]

    if M < 2:
        quality.trustworthy = False
        quality.messages.append("fewer than two strands survive; no braid.")
        empty = unroll(strands, k, np.arange(max(M, 1), dtype=int))
        return BraidTopology(
            lambda_val=lambda_val, unrolled=empty, crossings=[],
            unrolled_crossings=[], word=[], word_str="1",
            nu_total=0.0, pair_nu={}, ordered_nu={}, pair_windings=np.zeros((M, M)),
            linking=np.zeros((0, 0)), period_label="", cycle_type=(),
            paper_class=None, closure_label="no braid", quality=quality,
            strand_labels=labels,
        )

    # --- closing permutation from the open path, then close with it ----------
    perm = closing_permutation(strands)
    closed, k_closed, quality.closure_defect = close_loop(strands, k, perm)
    un = unroll(closed, k_closed, perm)

    # --- indices on the SINGLE closed loop -----------------------------------
    # Paper Eqs. 9-13 integrate over one Brillouin zone, and Table I tabulates
    # those one-loop values, so the indices must not be read off the unrolled
    # braid (which would multiply them by the number of laps).  Unrolling is for
    # drawing: it is what makes a 4π/6π component visibly close.
    windings = pair_matrix(closed)
    ordered_nu = {
        f"ν^({labels[i]},{labels[j]})": windings[i, j]
        for i, j in combinations(range(M), 2)
    }
    pair_nu = {
        f"ν_{{{labels[i]};{labels[j]}}}": 2.0 * windings[i, j]
        for i, j in combinations(range(M), 2)
    }
    nu = float(sum(2.0 * windings[i, j] for i, j in combinations(range(M), 2)))
    link = linking_matrix(closed, un.components)

    # --- crossings and canonical word ----------------------------------------
    # Word from one loop: its *closure* is the link, which is the object Table I
    # names.  Drawing crossings come from the unrolled braid (panel A).
    cross = crossings(closed, k_closed, proj_mix=proj_mix)
    word = canonical_word(word_from_crossings(cross))
    word_str = word_to_string(word)
    cross_unrolled = (cross if un.n_laps == 1
                      else crossings(un.strands, un.k, proj_mix=proj_mix))

    # --- quality --------------------------------------------------------------
    gaps = np.full(len(k_closed), np.inf)
    for i, j in combinations(range(M), 2):
        gaps = np.minimum(gaps, np.abs(closed[:, i] - closed[:, j]))
    i_min = int(np.argmin(gaps))
    quality.min_gap = float(gaps[i_min])
    quality.min_gap_k = float(k_closed[i_min])

    # Only the permutation-invariant combinations are quantized on one loop.  An
    # individual ν_{a;b} across a 4π component is legitimately a half-integer, so
    # checking it against an integer would flag correct physics as broken.
    quantized = [nu]
    quantized += [self_writhe(closed, c) for c in un.components if len(c) > 1]
    quantized += [nu_group(closed, a, b) for a, b in combinations(un.components, 2)]
    finite = [v for v in quantized if np.isfinite(v)]
    quality.integer_defect = float(
        max(abs(v - round(v)) for v in finite) if finite else np.nan
    )

    if not np.isfinite(quality.integer_defect) or quality.integer_defect > integer_tol:
        quality.trustworthy = False
        quality.messages.append(
            f"indices are {quality.integer_defect:.3f} from integer (tol {integer_tol:g}): "
            "the k-grid may be too coarse, or the path passes very near an "
            "exceptional point."
        )
    # ν is the exponent sum of the braid word; disagreement means the planar
    # projection and the winding integrals see different topologies.
    if word and abs(nu - exponent_sum(word)) > 0.5:
        quality.trustworthy = False
        quality.messages.append(
            f"ν from windings ({nu:+.2f}) disagrees with the braid word exponent "
            f"sum ({exponent_sum(word):+d}); crossing detection is unreliable here."
        )

    cycle_type = tuple(sorted((len(c) for c in un.components), reverse=True))
    paper_class, closure = classify(cycle_type, nu, pair_nu, word)
    if not quality.trustworthy:
        paper_class = None

    return BraidTopology(
        lambda_val=lambda_val,
        unrolled=un,
        crossings=cross,
        unrolled_crossings=cross_unrolled,
        word=word,
        word_str=word_str,
        nu_total=nu,
        pair_nu=pair_nu,
        ordered_nu=ordered_nu,
        pair_windings=windings,
        linking=link,
        period_label=period_label(un.components),
        cycle_type=cycle_type,
        paper_class=paper_class,
        closure_label=closure,
        quality=quality,
        strand_labels=labels,
    )


@dataclass
class BraidScan:
    """A :class:`BraidTopology` per λ, plus where the class changes."""

    lambda_values: np.ndarray
    topologies: list[BraidTopology]
    transition_lambdas: list[float]
    transition_labels: list[str]

    def at(self, lam) -> BraidTopology:
        """Nearest-λ lookup."""
        idx = int(np.argmin(np.abs(self.lambda_values - float(lam))))
        return self.topologies[idx]

    @property
    def nu_total(self) -> np.ndarray:
        return np.array([t.nu_total for t in self.topologies], dtype=float)

    @property
    def class_labels(self) -> list[str]:
        return [t.class_label for t in self.topologies]

    def most_interesting_lambda(self) -> float:
        """The λ worth showing as the default single-λ panel.

        Ranked by, in order: trustworthy first; then the most exotic
        periodicity (a 6π component beats a 4π one beats three 2π ones); then
        the largest |ν|; then the largest minimum gap, as a cleanliness
        tiebreak.

        Ranking by |ν| alone would be a trap: on noisy data the largest |ν|
        belongs to the slice where the strands nearly collide and the projection
        picks up spurious crossings — exactly the slice least worth showing.
        """
        if not self.topologies:
            return float("nan")

        def key(t: BraidTopology):
            return (
                1 if t.quality.trustworthy else 0,
                t.unrolled.n_laps,
                abs(t.nu_total) if np.isfinite(t.nu_total) else -1.0,
                t.quality.min_gap if np.isfinite(t.quality.min_gap) else -1.0,
            )

        return float(max(self.topologies, key=key).lambda_val)


def analyze_braid_scan(
    result,
    *,
    deriv_weight: float = 0.1,
    integer_tol: float = 5e-2,
    lambda_indices: Sequence[int] | None = None,
) -> BraidScan:
    """Run :func:`analyze_braid` across every λ of a ``KSpacePolesResult``.

    Strands come from ``kspace.plotting.track_permutation`` — continuity-sorted
    along k *and* matched across λ — so strand 0 is the same physical branch in
    every slice and the class strip is comparable from one λ to the next.

    Transitions are the λ where the class label changes; those are the
    exceptional points (paper §IV).
    """
    from mpm_analysis.kspace.plotting import track_permutation

    perm = track_permutation(result, deriv_weight=deriv_weight)
    poles = np.take_along_axis(np.asarray(result.poles, dtype=complex), perm, axis=2)
    k = np.asarray(result.k_array, dtype=float)

    idx = list(range(result.n_lambda)) if lambda_indices is None else list(lambda_indices)
    lam = np.asarray(result.lambda_values, dtype=float)[idx]

    topos = [
        analyze_braid(poles[li], k, lambda_val=float(result.lambda_values[li]),
                      integer_tol=integer_tol)
        for li in idx
    ]

    t_lam: list[float] = []
    t_lbl: list[str] = []
    for i in range(1, len(topos)):
        a, b = topos[i - 1], topos[i]
        if a.class_label != b.class_label:
            t_lam.append(float(0.5 * (lam[i - 1] + lam[i])))
            t_lbl.append(f"{a.class_label} → {b.class_label}")

    return BraidScan(lambda_values=lam, topologies=topos,
                     transition_lambdas=t_lam, transition_labels=t_lbl)


def print_braid_scan(scan: BraidScan) -> None:
    """ASCII summary table (safe for the Windows console codepage)."""
    def safe(text) -> str:
        return (str(text).replace("ν", "nu").replace("λ", "lambda")
                .replace("σ", "s").replace("π", "pi").replace("→", "->")
                .replace("⁻¹", "^-1").replace("Δ", "d")
                .encode("ascii", "replace").decode("ascii"))

    print("\n=== Braid topology (paper indices, unrolled to full period) ===")
    print(f"{'lambda':>8} {'nu':>5} {'period':>14} {'class':>7} {'word':<14} "
          f"{'min|d|':>9} {'ok':>3}")
    for t in scan.topologies:
        cls = t.paper_class or "-"
        print(f"{t.lambda_val:8.3f} {t.nu_total:+5.1f} {safe(t.period_label):>14} "
              f"{cls:>7} {safe(t.word_str):<14} {t.quality.min_gap:9.3g} "
              f"{'y' if t.quality.trustworthy else 'N':>3}")
    if scan.transition_lambdas:
        print("  transitions (exceptional points):")
        for lm, lbl in zip(scan.transition_lambdas, scan.transition_labels):
            print(f"    lambda~{lm:.3f}:  {safe(lbl)}")
    else:
        print("  no class transitions over this lambda range.")
