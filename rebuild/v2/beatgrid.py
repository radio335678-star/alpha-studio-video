"""
beatgrid.py -- musical time base for the 25 s Alpha promo.

EVERYTHING in this module was measured from audio.wav by reanalyze.py +
reanalyze2.py (see the sibling files).  Nothing here is hand-tuned; the module
is intentionally *static* -- it does not load or re-analyse audio at import
time, so importing it is instant and fully deterministic.

How the grid was obtained
-------------------------
1.  Onset-strength envelope (hop 512 @ 48 kHz -> 93.75 Hz grid), smoothed.
2.  Joint BPM + phase fit: scan 115..125 BPM in 0.005 steps x 400 phases,
    scoring the mean envelope value sampled exactly on the beat lattice.  The
    optimum was 119.975 BPM with a phase offset of 0.0563 s.  A sweep of
    per-segment tempos shows 119.65..120.15 BPM in every dense 4 s window, and
    a linear-drift refit over strong onsets improves the residual by 0.7 ms
    (i.e. not at all) -> the grid is FLAT, no drift compensation needed.
    NB: the widely-quoted 119.68 BPM is a coarse librosa tempogram
    quantisation; on the same scoring function the measured grid scores 0.386
    vs 0.211, i.e. ~1.8x better, and 46% of onsets land within 40 ms of it
    versus 41% for the 119.68 grid.  119.975 BPM is used here.
3.  Meter: energy in the 40-140 Hz kick band was summed on each of the four
    beat phases.  Phase 0 scored 0.378 / 0.259 / 0.369 / 0.264 -> the bar
    starts on phase 0, so downbeats are beats 0, 4, 8, ...
4.  Frame alignment: the period is 0.500104 s = 15.0031 frames, so beats land
    on frames 2 + 15n and downbeats on frames 2 + 60n.  Every musical event
    is within 11 ms of its frame, so frame-locked cuts are sample-tight.
5.  Hits / accents: onset peaks (>=300 ms apart, >=10% prominence) were scored
    0.45*prominence + 0.30*local-salience + 0.25*abs-level.  "Local salience"
    is (peak - local median)/local MAD over a +/-1.6 s window, which correctly
    surfaces percussive hits inside the quiet outro (20.32 s and 21.89 s rank
    on salience alone even though the outro RMS is low).
6.  Energy: per-frame RMS resampled to 750 values, gaussian-smoothed
    (sigma=2 frames) and peak-normalised to 0..1.
7.  Sections: novelty peaks in the smoothed-energy curve (5.63 / 13.80 /
    18.06 s), each snapped to the nearest downbeat so section_at() is
    bar-exact.
"""

from __future__ import annotations

import bisect

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a hard dep of the renderer anyway
    _np = None


# --------------------------------------------------------------------------
# SNAP -- every tunable constant lives here and nowhere else.
# --------------------------------------------------------------------------
SNAP = {
    # --- transport -------------------------------------------------------
    "FPS": 30,
    "TOTAL_FRAMES": 750,
    "DURATION": 25.0,

    # --- tempo grid (measured, see module docstring) --------------------
    "BPM": 119.975,
    "PERIOD": 0.500104,
    "PHASE0": 0.056262,
    "BEATS_PER_BAR": 4,
    "BEATS": 50,
    "DOWNBEAT_PHASE": 0,
    "FRAMES_PER_BEAT": 15.0031,
    "FRAMES_PER_BAR": 60.0125,
    # offset of the beat lattice in frames: beat i sits on BEAT_FRAME0 + i*F_BEAT
    "BEAT_FRAME0": 2,
    "BAR_FRAME0": 2,

    # --- accent detection thresholds ------------------------------------
    "HIT_MIN_SEP_S": 0.30,
    "HIT_MIN_PROM_FRAC": 0.10,
    "HIT_SCORE_MIN": 0.5,
    "ACCENT_SCORE_MIN": 0.55,
    # accent score weights (must sum to 1.0)
    "W_PROM": 0.45,
    "W_SALIENCE": 0.30,
    "W_LEVEL": 0.25,
    "SALIENCE_WINDOW_S": 1.6,

    # --- pulse shapes (frames) ------------------------------------------
    # A pulse is (1 - d/decay)**2 for 0 <= d < decay, else 0: 1.0 exactly on
    # the event frame, smooth, and exactly zero at the decay edge.
    "HIT_DECAY": 7.0,
    "BEAT_DECAY": 8.0,
    "BAR_DECAY": 12.0,

    # --- reveal staggering ----------------------------------------------
    # default number of beats between staggered line reveals
    "REVEAL_SPACING_BEATS": 1,

    # --- sections --------------------------------------------------------
    "SECTION_LABELS": ['intro', 'build', 'drop', 'outro'],
}

FPS = SNAP["FPS"]
TOTAL_FRAMES = SNAP["TOTAL_FRAMES"]
DURATION = SNAP["DURATION"]
PERIOD = SNAP["PERIOD"]
PHASE0 = SNAP["PHASE0"]
BEATS_PER_BAR = SNAP["BEATS_PER_BAR"]


# --------------------------------------------------------------------------
# Raw measured data
# --------------------------------------------------------------------------
BEAT_TIMES = (
    0.0563, 0.5564, 1.0565, 1.5566, 2.0567, 2.5568, 3.0569, 3.557, 4.0571, 4.5572,
    5.0573, 5.5574, 6.0575, 6.5576, 7.0577, 7.5578, 8.0579, 8.558, 9.0581, 9.5582,
    10.0583, 10.5584, 11.0586, 11.5587, 12.0588, 12.5589, 13.059, 13.5591, 14.0592,
    14.5593, 15.0594, 15.5595, 16.0596, 16.5597, 17.0598, 17.5599, 18.06, 18.5601,
    19.0602, 19.5603, 20.0604, 20.5605, 21.0606, 21.5607, 22.0608, 22.561, 23.0611,
    23.5612, 24.0613, 24.5614
)

DOWNBEAT_TIMES = (
    0.0563, 2.0567, 4.0571, 6.0575, 8.0579, 10.0583, 12.0588, 14.0592, 16.0596, 18.06,
    20.0604, 22.0608, 24.0613
)

HIT_TIMES = (
    0.0533, 3.0507, 4.0533, 5.056, 7.5627, 8.5653, 9.5573, 10.56, 11.5627, 12.5547,
    16.064, 16.5547, 18.0587, 18.56, 19.0613, 19.5627, 20.32, 21.888, 22.624
)

HIT_SCORES = (
    0.9973, 0.8143, 0.8204, 0.8644, 0.5585, 0.5217, 0.6937, 0.6876, 0.7157, 0.6527,
    0.8547, 0.7793, 0.7149, 0.7989, 0.5963, 0.7861, 0.555, 0.5646, 0.504
)

ACCENT_TIMES = (
    0.0533, 3.0507, 4.0533, 5.056, 7.5627, 9.5573, 10.56, 11.5627, 12.5547, 16.064,
    16.5547, 18.0587, 18.56, 19.0613, 19.5627, 20.32, 21.888
)

# 750-value smoothed RMS envelope, normalised 0..1, one value per output frame
ENERGY = (
    0.07234, 0.09004, 0.11998, 0.1551, 0.19048, 0.22407, 0.25578, 0.28668, 0.3185,
    0.35231, 0.38711, 0.4196, 0.44531, 0.4605, 0.46318, 0.45337, 0.43302, 0.4052, 0.37263,
    0.3367, 0.29843, 0.26018, 0.22579, 0.19875, 0.18022, 0.16861, 0.16084, 0.15425,
    0.14769, 0.14131, 0.13567, 0.13105, 0.12735, 0.12415, 0.12097, 0.11749, 0.11358,
    0.10917, 0.10433, 0.09937, 0.09486, 0.09153, 0.0902, 0.09151, 0.09581, 0.103, 0.11245,
    0.12311, 0.13381, 0.1437, 0.15244, 0.16006, 0.16665, 0.17213, 0.17646, 0.18002,
    0.1846, 0.1951, 0.22057, 0.27103, 0.3486, 0.43923, 0.51553, 0.55252, 0.54262, 0.49718,
    0.43644, 0.37849, 0.33338, 0.3018, 0.27744, 0.25148, 0.21777, 0.17621, 0.13276,
    0.09561, 0.07051, 0.05832, 0.05577, 0.05801, 0.06193, 0.06849, 0.08163, 0.10331,
    0.1293, 0.15085, 0.16177, 0.16547, 0.17731, 0.21874, 0.30185, 0.4136, 0.51733,
    0.57526, 0.57263, 0.52243, 0.45154, 0.38363, 0.33039, 0.29178, 0.26124, 0.2314,
    0.19753, 0.15884, 0.11843, 0.08178, 0.0538, 0.0362, 0.02742, 0.02507, 0.02873,
    0.04115, 0.0662, 0.10421, 0.14821, 0.18583, 0.20704, 0.21223, 0.21595, 0.24209,
    0.30862, 0.40929, 0.51047, 0.5717, 0.57341, 0.52665, 0.45936, 0.39577, 0.34579,
    0.30723, 0.27317, 0.23767, 0.19813, 0.15557, 0.11404, 0.07881, 0.05356, 0.03863,
    0.03167, 0.02983, 0.03159, 0.03737, 0.04831, 0.06401, 0.08093, 0.09393, 0.10087,
    0.10713, 0.12682, 0.17681, 0.26298, 0.36829, 0.45807, 0.50255, 0.49526, 0.45128,
    0.39264, 0.33674, 0.29246, 0.26073, 0.23666, 0.2134, 0.18674, 0.15827, 0.13415,
    0.11966, 0.11415, 0.11228, 0.11027, 0.10955, 0.11412, 0.12601, 0.14391, 0.16468,
    0.18516, 0.20325, 0.2195, 0.24007, 0.27988, 0.35941, 0.49021, 0.65806, 0.82315,
    0.94348, 1.0, 0.9996, 0.95804, 0.88544, 0.78851, 0.68092, 0.58475, 0.51938, 0.49043,
    0.49055, 0.50905, 0.53851, 0.57242, 0.60047, 0.6098, 0.59087, 0.54246, 0.47473,
    0.40804, 0.3628, 0.34514, 0.34362, 0.34145, 0.33207, 0.32449, 0.33632, 0.37943,
    0.44714, 0.51435, 0.55291, 0.5489, 0.50719, 0.44455, 0.3805, 0.32999, 0.29816,
    0.28024, 0.26793, 0.25801, 0.25713, 0.27888, 0.33241, 0.40957, 0.48396, 0.52675,
    0.52571, 0.48966, 0.43794, 0.38789, 0.34912, 0.32375, 0.30886, 0.29982, 0.29404,
    0.29467, 0.31119, 0.35237, 0.41441, 0.47686, 0.51419, 0.51298, 0.47807, 0.42524,
    0.37055, 0.32437, 0.29026, 0.26665, 0.25021, 0.2403, 0.24237, 0.26681, 0.32009,
    0.39348, 0.46218, 0.50014, 0.49722, 0.46234, 0.41312, 0.36495, 0.32642, 0.29955,
    0.28174, 0.26893, 0.25952, 0.25796, 0.2746, 0.31789, 0.38272, 0.44755, 0.48718,
    0.48922, 0.45879, 0.41069, 0.35956, 0.31441, 0.27775, 0.24793, 0.22327, 0.20584,
    0.20336, 0.2269, 0.28242, 0.36045, 0.43559, 0.48071, 0.48373, 0.45189, 0.40256,
    0.35181, 0.30872, 0.27555, 0.25078, 0.23245, 0.22131, 0.22294, 0.2463, 0.29653,
    0.36585, 0.43277, 0.47352, 0.47641, 0.44599, 0.39645, 0.34272, 0.295, 0.25718,
    0.22857, 0.2075, 0.19523, 0.19788, 0.22465, 0.2804, 0.35647, 0.42975, 0.47482,
    0.47912, 0.44796, 0.39748, 0.34434, 0.29897, 0.26417, 0.23772, 0.21688, 0.20275,
    0.20267, 0.22804, 0.28537, 0.36514, 0.44095, 0.48462, 0.48443, 0.44966, 0.40041,
    0.35505, 0.32347, 0.3061, 0.29689, 0.28872, 0.2789, 0.27248, 0.28125, 0.31558,
    0.37222, 0.4306, 0.46539, 0.46381, 0.43092, 0.38128, 0.32844, 0.28056, 0.2408,
    0.20921, 0.18518, 0.1704, 0.17126, 0.19748, 0.25421, 0.33206, 0.40672, 0.45259,
    0.4584, 0.43054, 0.38408, 0.33309, 0.28665, 0.24918, 0.22203, 0.20543, 0.20032,
    0.21027, 0.24147, 0.29724, 0.36914, 0.43532, 0.47227, 0.47027, 0.43755, 0.39159,
    0.3481, 0.31537, 0.29436, 0.28177, 0.27387, 0.26991, 0.27472, 0.2979, 0.34559,
    0.40909, 0.46441, 0.48815, 0.47388, 0.43248, 0.38049, 0.3306, 0.28937, 0.25814,
    0.23503, 0.2179, 0.2077, 0.21049, 0.23576, 0.28868, 0.36037, 0.42737, 0.46516,
    0.46362, 0.42976, 0.37879, 0.32477, 0.27663, 0.23827, 0.21054, 0.19383, 0.19018,
    0.20416, 0.24172, 0.30487, 0.38486, 0.46186, 0.51537, 0.53613, 0.52783, 0.4987,
    0.45402, 0.3974, 0.33664, 0.28408, 0.25046, 0.23995, 0.25181, 0.28444, 0.3352,
    0.39603, 0.4523, 0.48926, 0.50004, 0.48679, 0.45552, 0.41135, 0.35803, 0.30034,
    0.24617, 0.20624, 0.19095, 0.2067, 0.25399, 0.32631, 0.40936, 0.48372, 0.53299,
    0.55098, 0.54125, 0.51091, 0.46581, 0.41015, 0.34877, 0.2896, 0.24394, 0.22297,
    0.23277, 0.27176, 0.33116, 0.39644, 0.45021, 0.47781, 0.47236, 0.43613, 0.37986,
    0.32065, 0.27469, 0.24752, 0.23207, 0.21791, 0.20275, 0.19627, 0.2137, 0.26328,
    0.33582, 0.40633, 0.44899, 0.45243, 0.4225, 0.37421, 0.32284, 0.27891, 0.2458,
    0.22067, 0.1992, 0.18171, 0.17623, 0.19528, 0.24565, 0.31797, 0.38772, 0.43027,
    0.43585, 0.41122, 0.37081, 0.32816, 0.29233, 0.26691, 0.25078, 0.24137, 0.23944,
    0.25249, 0.29274, 0.36746, 0.46711, 0.56422, 0.62872, 0.64632, 0.6228, 0.57413,
    0.51533, 0.45579, 0.39971, 0.34832, 0.3029, 0.26816, 0.25483, 0.27741, 0.34389,
    0.44301, 0.54384, 0.61291, 0.63385, 0.61217, 0.56576, 0.51311, 0.46621, 0.42869,
    0.39794, 0.36982, 0.34363, 0.32624, 0.33225, 0.37488, 0.45097, 0.53606, 0.59859,
    0.61994, 0.60139, 0.55672, 0.50186, 0.44845, 0.40154, 0.36064, 0.32325, 0.28908,
    0.26323, 0.25585, 0.27539, 0.31822, 0.36577, 0.39537, 0.39525, 0.36928, 0.33059,
    0.29245, 0.26259, 0.2416, 0.22535, 0.20981, 0.19539, 0.18878, 0.20062, 0.23794,
    0.29506, 0.35256, 0.38852, 0.39239, 0.3691, 0.33276, 0.2976, 0.27194, 0.25634,
    0.24576, 0.23461, 0.22185, 0.21383, 0.2225, 0.25699, 0.31249, 0.36862, 0.40261,
    0.40567, 0.38749, 0.36721, 0.36029, 0.36946, 0.3842, 0.38884, 0.3743, 0.34452,
    0.31356, 0.29662, 0.30093, 0.32122, 0.34297, 0.35188, 0.34249, 0.31957, 0.29416,
    0.27871, 0.28152, 0.30093, 0.32481, 0.33841, 0.33422, 0.31536, 0.29239, 0.2779,
    0.28052, 0.29899, 0.32164, 0.33384, 0.32736, 0.30307, 0.26698, 0.2258, 0.18506,
    0.14866, 0.11857, 0.09501, 0.07719, 0.06399, 0.05449, 0.04812, 0.04451, 0.04341,
    0.04447, 0.04727, 0.05135, 0.0562, 0.06122, 0.06562, 0.06848, 0.06887, 0.06593,
    0.05925, 0.04952, 0.03869, 0.02903, 0.02191, 0.01741, 0.01475, 0.01323, 0.01276,
    0.0141, 0.01871, 0.02777, 0.04033, 0.05235, 0.05866, 0.05678, 0.04865, 0.03848,
    0.02953, 0.02293, 0.01853, 0.0161, 0.01582, 0.01788, 0.02162, 0.0252, 0.0265, 0.02464,
    0.02062, 0.0163, 0.01333, 0.01301, 0.01672, 0.02542, 0.03798, 0.05024, 0.05718,
    0.05695, 0.0525, 0.04876, 0.04828, 0.04958, 0.0492, 0.04505, 0.03785, 0.02987,
    0.02297, 0.01786, 0.01449, 0.01272, 0.0126, 0.0144, 0.0184, 0.02467, 0.03286, 0.04224,
    0.05189, 0.0609, 0.06853, 0.07426, 0.07817, 0.08223, 0.09252, 0.12069, 0.18055,
    0.27772, 0.39907, 0.51491, 0.59625, 0.63163, 0.62794, 0.5978, 0.54907, 0.48646,
    0.41884, 0.35981, 0.31983, 0.29953, 0.29106, 0.28478, 0.27477, 0.26028, 0.24359,
    0.22729, 0.2128, 0.20048, 0.19011, 0.18122, 0.17284, 0.16342, 0.15151, 0.13681,
    0.12054, 0.10453, 0.09003, 0.07741, 0.0665, 0.05701, 0.04873, 0.04148, 0.03506,
    0.02929, 0.02407, 0.01942, 0.01545, 0.01248, 0.01086
)

# mean ENERGY over each beat's own span (index-aligned with BEAT_TIMES)
BEAT_ENERGY = (
    0.331, 0.22115, 0.10638, 0.2045, 0.31865, 0.14709, 0.31653, 0.15055, 0.32009, 0.10661,
    0.29018, 0.23442, 0.71895, 0.42926, 0.37738, 0.38668, 0.35816, 0.35795, 0.33302,
    0.33511, 0.32072, 0.32501, 0.35672, 0.30277, 0.31648, 0.35451, 0.3249, 0.31564,
    0.38466, 0.35289, 0.38888, 0.31474, 0.29925, 0.33564, 0.44304, 0.46482, 0.42335,
    0.28039, 0.29128, 0.35904, 0.31172, 0.19125, 0.05292, 0.03103, 0.01964, 0.04125,
    0.05247, 0.4367, 0.195, 0.0406
)

SECTIONS = [{'start_frame': 0, 'end_frame': 182, 'start': 0.0, 'end': 6.0667, 'label': 'intro'}, {'start_frame': 182, 'end_frame': 422, 'start': 6.0667, 'end': 14.0667, 'label': 'build'}, {'start_frame': 422, 'end_frame': 542, 'start': 14.0667, 'end': 18.0667, 'label': 'drop'}, {'start_frame': 542, 'end_frame': 750, 'start': 18.0667, 'end': 25.0, 'label': 'outro'}]

# --------------------------------------------------------------------------
# Derived frame-space tables (the renderer's working units)
# --------------------------------------------------------------------------
BEAT_FRAMES = tuple(round(t * FPS) for t in BEAT_TIMES)
DOWNBEAT_FRAMES = tuple(round(t * FPS) for t in DOWNBEAT_TIMES)
HIT_FRAMES = tuple(round(t * FPS) for t in HIT_TIMES)
ACCENT_FRAMES = tuple(round(t * FPS) for t in ACCENT_TIMES)

#: frames that should trigger a visible "accent" (flash / punch / scale pop).
#: 17 events, roughly one every 1.5 s across the 25 s piece.
accent_frames = list(ACCENT_FRAMES)

_BAR_FRAMES = SNAP["BAR_FRAME0"] + SNAP["FRAMES_PER_BAR"] * SNAP["BEATS_PER_BAR"]
_BEAT_OFF = SNAP["BEAT_FRAME0"]


def _is_scalar(x):
    """True for numbers / 0-d arrays, False for anything iterable."""
    if isinstance(x, (list, tuple, range, set)):
        return False
    if _np is not None and isinstance(x, _np.ndarray):
        return x.ndim == 0
    return not hasattr(x, "__iter__")


def _as_list(x):
    """Materialise any non-scalar input (array, range, generator, ...) to a list."""
    if _np is not None and isinstance(x, _np.ndarray):
        return x.ravel().tolist()
    if isinstance(x, (list, tuple, range, set)):
        return list(x)
    if hasattr(x, "__iter__"):
        return list(x)
    return [x]


# --------------------------------------------------------------------------
# Phase / index queries
# --------------------------------------------------------------------------
def beat_times(t=None, window=None):
    """Beat times in the neighbourhood of *t*.

    Parameters
    ----------
    t : float | sequence[float] | None
        A time in seconds, a sequence of times, or None for the whole grid.
    window : float | None
        Half-width in seconds.  Defaults to one beat period.  For a sequence
        *t* the window is taken around [min(t), max(t)].
    """
    if t is None:
        return list(BEAT_TIMES)
    if isinstance(t, (list, tuple)) or (_np is not None and isinstance(t, _np.ndarray)):
        vals = _as_list(t)
        lo, hi = min(vals), max(vals)
        if window is None:
            window = PERIOD
        lo -= window
        hi += window
    else:
        w = PERIOD if window is None else window
        lo, hi = float(t) - w, float(t) + w
    i = bisect.bisect_left(BEAT_TIMES, lo)
    out = []
    while i < len(BEAT_TIMES) and BEAT_TIMES[i] <= hi:
        out.append(BEAT_TIMES[i])
        i += 1
    return out


def beat_index(t):
    """Index of the beat containing *t* (0 at the first beat; may be negative)."""
    if _np is None or _is_scalar(t):
        return int((float(t) - PHASE0) // PERIOD)
    return _np.floor((_np.asarray(t, dtype=float) - PHASE0) / PERIOD).astype(int)


def beat_phase(t):
    """Position inside the current beat, 0.0 at the beat, ~1.0 just before the next.

    Always in [0, 1).  Accepts a scalar or any array-like.
    """
    if _np is None or _is_scalar(t):
        return float(((float(t) - PHASE0) / PERIOD) % 1.0)
    return ((_np.asarray(t, dtype=float) - PHASE0) / PERIOD) % 1.0


def downbeat_phase(t):
    """Position inside the current 4/4 bar, 0.0 on the downbeat, [0, 1)."""
    bar = PERIOD * BEATS_PER_BAR
    if _np is None or _is_scalar(t):
        return float(((float(t) - PHASE0) / bar) % 1.0)
    return ((_np.asarray(t, dtype=float) - PHASE0) / bar) % 1.0


def bar_index(t):
    """Index of the bar containing *t* (bar 0 contains the first beat)."""
    return int((float(t) - PHASE0) // (PERIOD * BEATS_PER_BAR))


# --------------------------------------------------------------------------
# Pulses -- all take frames, return 0..1
# --------------------------------------------------------------------------
def _pulse_scalar(f, frames, decay):
    """(1 - d/decay)**2 for 0 <= d < decay since the last event, else 0."""
    f = int(round(float(f)))
    i = bisect.bisect_right(frames, f) - 1
    if i < 0:
        return 0.0
    d = f - frames[i]
    if d >= decay:
        return 0.0
    u = 1.0 - d / decay
    return u * u


def _pulse(f, frames, decay):
    if _np is None or _is_scalar(f):
        return _pulse_scalar(f, frames, decay)
    return _np.array([_pulse_scalar(v, frames, decay) for v in _as_list(f)],
                     dtype=float)


def hit_pulse(f, decay=None):
    """Percussive accent hit flash.  1.0 on the hit frame, decays over SNAP['HIT_DECAY']."""
    d = SNAP["HIT_DECAY"] if decay is None else decay
    return _pulse(f, HIT_FRAMES, d)


def beat_pulse(f, decay=None):
    """Every-beat pulse.  Decays over SNAP['BEAT_DECAY'] frames."""
    d = SNAP["BEAT_DECAY"] if decay is None else decay
    return _pulse(f, BEAT_FRAMES, d)


def bar_pulse(f, decay=None):
    """Downbeat (bar-line) pulse -- the heaviest accent in the piece."""
    d = SNAP["BAR_DECAY"] if decay is None else decay
    return _pulse(f, DOWNBEAT_FRAMES, d)


# --------------------------------------------------------------------------
# Beat-locked staggering for reveals
# --------------------------------------------------------------------------
def snapped_onsets(local_f, spacing=1, limit=None, count=None):
    """Frame *offsets* (relative to *local_f*) for beat-locked staggered reveals.

    The first reveal is the first beat frame at or after *local_f* (so a cut
    that already lands on a beat reveals immediately with offset 0), and each
    subsequent reveal is *spacing* beats later.  Because the beat lattice is
    ``2 + 15n`` frames, reveals land on frames 2, 17, 32, ... exactly -- no
    sub-frame drift, ever.

    Parameters
    ----------
    local_f : int      scene start frame
    spacing : int      beats between reveals (>= 1)
    limit   : int      last frame to emit (default: end of timeline)
    count   : int      max number of offsets to emit (default: unlimited)
    """
    spacing = max(1, int(spacing))
    step = spacing * int(round(PERIOD * FPS))
    last = (TOTAL_FRAMES - 1) if limit is None else int(limit)
    first = _BEAT_OFF + int(_np.ceil((int(local_f) - _BEAT_OFF) / step)) * step
    out = []
    f = first
    while f <= last:
        out.append(f - int(local_f))
        if count is not None and len(out) >= int(count):
            break
        f += step
    return out


# --------------------------------------------------------------------------
# Energy / sections
# --------------------------------------------------------------------------
def energy_at(f):
    """Smoothed 0..1 loudness at frame *f* (clamped to the 750-value curve)."""
    if _np is None or _is_scalar(f):
        x = float(f)
        if x <= 0:
            return ENERGY[0]
        if x >= TOTAL_FRAMES - 1:
            return ENERGY[-1]
        i = int(x)
        w = x - i
        return ENERGY[i] * (1.0 - w) + ENERGY[i + 1] * w
    idx = [min(max(float(v), 0.0), TOTAL_FRAMES - 1) for v in _as_list(f)]
    return _np.interp(idx, list(range(TOTAL_FRAMES)), ENERGY)


def energy_mean(f0, f1):
    """Mean smoothed energy across the frame range [f0, f1)."""
    a = max(0, int(round(f0)))
    b = min(TOTAL_FRAMES, int(round(f1)))
    if b <= a:
        return energy_at(a)
    return sum(ENERGY[a:b]) / (b - a)


def beats_in(f0, f1):
    """Number of beat times falling in [f0, f1) -- i.e. how many beats a scene spans."""
    a = f0 / FPS
    b = f1 / FPS
    i = bisect.bisect_left(BEAT_TIMES, a - 1e-9)
    n = 0
    while i < len(BEAT_TIMES) and BEAT_TIMES[i] < b - 1e-9:
        n += 1
        i += 1
    return n


def section_at(t):
    """Section label ('intro' | 'build' | 'drop' | 'outro') containing time *t*."""
    fr = t * FPS if t < TOTAL_FRAMES else TOTAL_FRAMES
    for s in SECTIONS:
        if s["start_frame"] <= fr < s["end_frame"]:
            return s["label"]
    return SECTIONS[-1]["label"]


def snap_to_event(frame, pool="all"):
    """Snap a frame to the nearest musical event; returns (frame, kind, time_s).

    pool: 'downbeat' | 'hit' | 'accent' | 'beat' | 'all'
    """
    table = {
        "downbeat": DOWNBEAT_FRAMES,
        "hit": HIT_FRAMES,
        "accent": ACCENT_FRAMES,
        "beat": BEAT_FRAMES,
    }.get(pool)
    if table is None:
        table = tuple(sorted(set(DOWNBEAT_FRAMES) | set(HIT_FRAMES) | set(BEAT_FRAMES)))
    frame = int(frame)
    best = min(table, key=lambda f: abs(f - frame))
    if best in DOWNBEAT_FRAMES:
        kind = "downbeat"
    elif best in HIT_FRAMES:
        kind = "hit"
    else:
        kind = "beat"
    return best, kind, round(best / FPS, 4)


__all__ = [
    "SNAP", "FPS", "TOTAL_FRAMES", "DURATION", "PERIOD", "PHASE0",
    "BEAT_TIMES", "DOWNBEAT_TIMES", "HIT_TIMES", "ACCENT_TIMES",
    "ENERGY", "BEAT_ENERGY", "SECTIONS",
    "BEAT_FRAMES", "DOWNBEAT_FRAMES", "HIT_FRAMES", "ACCENT_FRAMES",
    "accent_frames",
    "beat_times", "beat_phase", "beat_index", "downbeat_phase", "bar_index",
    "hit_pulse", "beat_pulse", "bar_pulse", "snapped_onsets",
    "energy_at", "energy_mean", "beats_in", "section_at", "snap_to_event",
]
