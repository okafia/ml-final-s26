"""
GTO-inspired track selection for the AutoDJ system.

Implements the methodology from the paper:
  - K-Means (k=3) clusters tracks into low/mid/high energy groups using
    tempo, loudness proxy (-replaygain), and spectral style (theme_descriptor).
  - KNN retrieves the 5 nearest candidates from the unplayed library by
    Euclidean distance in standardised feature space.
  - A rule-based GTO policy (Call / Hold / Raise) picks the action that
    minimises deviation from a pre-planned sigmoid energy arc running from
    the opening track's energy to the library's peak energy.

All features are read from the annotation JSON files that already exist on
disk; no additional audio processing is required.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import songcollection as sc_module

import logging
logger = logging.getLogger('colorlogger')

# Maximum Camelot-Wheel distance allowed for the Hold (harmonic bridge) action.
CAMELOT_MAX_DISTANCE = 1


def camelot_distance(key1, scale1, key2, scale2):
    """Return the absolute Camelot-Wheel distance (0–6) between two keys."""
    return abs(sc_module.distance_keys_circle_of_fifths(key1, scale1, key2, scale2))


class GTOMixer:
    """
    GTO-inspired track selector combining K-Means clustering, KNN retrieval,
    and a deterministic energy-arc policy.

    Cluster energy labels
    ---------------------
    0 = low-energy   (maps to GTO "Call"  — sustain)
    1 = mid-energy   (maps to GTO "Hold"  — harmonic bridge)
    2 = high-energy  (maps to GTO "Raise" — peak drop)
    """

    GTO_CALL  = 0   # Sustain: stay in same energy cluster
    GTO_HOLD  = 1   # Harmonic bridge: Camelot distance ≤ 1
    GTO_RAISE = 2   # Peak drop: select from highest-energy cluster

    _ACTION_NAMES = ['CALL', 'HOLD', 'RAISE']

    def __init__(self, n_clusters=3, n_neighbors=5):
        self.n_clusters  = n_clusters
        self.n_neighbors = n_neighbors

        self.kmeans  = None
        self.scaler  = StandardScaler()
        self.fitted  = False

        # Caches populated by fit() so songs need not be re-opened later.
        self.song_raw_features    = {}  # title -> [tempo, energy, spectral]
        self.song_scaled_features = {}  # title -> scaled feature vector (np.ndarray)
        self.song_clusters        = {}  # title -> energy-ordered cluster (0/1/2)
        self.song_keys            = {}  # title -> (key, scale)
        self.cluster_energy_order = {}  # raw KMeans cluster id -> energy rank

        # Energy-arc state
        self.mix_step   = 0
        self.energy_arc = np.array([])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_raw(self, song):
        """
        Return [tempo, -replaygain, theme_descriptor[0]] from an already-open song.

        Feature choices (all from existing annotations):
          tempo          — BPM; captures rhythmic energy
          -replaygain    — loudness proxy; louder tracks ↔ higher energy
          theme_descriptor[0] — first PCA component of spectral contrast;
                                captures brightness / timbre
        """
        tempo    = song.tempo    if song.tempo    is not None else 170.0
        energy   = -song.replaygain if song.replaygain is not None else 0.0
        td       = getattr(song, 'song_theme_descriptor', None)
        spectral = float(td[0]) if td else 0.0
        return [tempo, energy, spectral]

    def _scaled_for(self, song):
        """Cached or freshly computed scaled feature vector (never opens/closes song)."""
        if song.title in self.song_scaled_features:
            return self.song_scaled_features[song.title]
        # Song was not in the original fitted set; compute on-the-fly.
        raw = np.array([self.song_raw_features.get(
            song.title, [170.0, 0.0, 0.0]
        )], dtype=float)
        vec = self.scaler.transform(raw)[0]
        self.song_scaled_features[song.title] = vec
        return vec

    # ------------------------------------------------------------------
    # Fitting (K-Means + StandardScaler)
    # ------------------------------------------------------------------

    def fit(self, songs):
        """
        Fit K-Means on the annotated library and cache every song's features,
        cluster label, and musical key.  Songs are opened then immediately
        closed so only lightweight JSON data is loaded.
        """
        valid_songs, raw_rows = [], []

        for s in songs:
            s.open()
            if s.tempo is None or s.replaygain is None:
                s.close()
                continue
            raw = self._extract_raw(s)
            self.song_raw_features[s.title] = raw
            self.song_keys[s.title]         = (s.key, s.scale)
            valid_songs.append(s)
            raw_rows.append(raw)
            s.close()

        if not valid_songs:
            logger.warning('GTOMixer.fit: no valid annotated songs found')
            return

        raw_array    = np.array(raw_rows, dtype=float)
        scaled_array = self.scaler.fit_transform(raw_array)

        for i, s in enumerate(valid_songs):
            self.song_scaled_features[s.title] = scaled_array[i]

        k = min(self.n_clusters, len(valid_songs))
        self.kmeans       = KMeans(n_clusters=k, random_state=42, n_init=10)
        raw_cluster_ids   = self.kmeans.fit_predict(scaled_array)

        # Map raw KMeans cluster ids to energy-ordered labels (0=low … k-1=high).
        cluster_energies = {}
        for i, c in enumerate(raw_cluster_ids):
            cluster_energies.setdefault(c, []).append(raw_array[i, 1])
        sorted_clusters = sorted(
            cluster_energies.keys(),
            key=lambda c: np.median(cluster_energies[c])
        )
        self.cluster_energy_order = {sorted_clusters[i]: i for i in range(len(sorted_clusters))}

        for i, s in enumerate(valid_songs):
            self.song_clusters[s.title] = self.cluster_energy_order[raw_cluster_ids[i]]

        sizes = {v: sum(1 for c in self.song_clusters.values() if c == v)
                 for v in range(k)}
        logger.info('GTOMixer: fitted on %d songs | cluster sizes %s', len(valid_songs), sizes)
        self.fitted = True

    # ------------------------------------------------------------------
    # Energy-arc planning
    # ------------------------------------------------------------------

    def plan_energy_arc(self, opening_song, n_steps):
        """
        Build a sigmoid energy arc over n_steps tracks.

        The arc starts at the opening song's loudness and rises toward the
        maximum loudness in the library, following a sigmoid curve as proposed
        in Flexer et al.'s energy-arc framework.
        """
        start = self.song_raw_features.get(opening_song.title, [170.0, 0.0, 0.0])[1]
        all_e = [f[1] for f in self.song_raw_features.values()]
        end   = max(all_e) if all_e else start + 5.0

        x               = np.linspace(-6, 6, max(n_steps, 2))
        sig             = 1.0 / (1.0 + np.exp(-x))
        self.energy_arc = start + (end - start) * sig
        self.mix_step   = 0

        logger.info('GTOMixer: energy arc %.2f → %.2f over %d steps', start, end, n_steps)

    # ------------------------------------------------------------------
    # Cluster assignment
    # ------------------------------------------------------------------

    def get_cluster(self, song):
        """Return the energy-ordered cluster label (0=low, 1=mid, 2=high)."""
        if song.title in self.song_clusters:
            return self.song_clusters[song.title]
        if not self.fitted:
            return 1
        raw_c = self.kmeans.predict(self._scaled_for(song).reshape(1, -1))[0]
        label = self.cluster_energy_order.get(raw_c, 1)
        self.song_clusters[song.title] = label
        return label

    # ------------------------------------------------------------------
    # KNN retrieval
    # ------------------------------------------------------------------

    def get_knn_candidates(self, current_song, candidate_songs):
        """
        Return up to n_neighbors candidates ordered by ascending Euclidean
        distance in the standardised feature space.

        All feature lookups use the cache populated during fit(); songs are
        not opened here so caller-managed open/close state is unaffected.
        """
        if not candidate_songs:
            return []

        cur_vec = self._scaled_for(current_song)

        valid, feats = [], []
        for s in candidate_songs:
            vec = self._scaled_for(s)
            valid.append(s)
            feats.append(vec)

        if not valid:
            return list(candidate_songs)[:self.n_neighbors]

        dists   = np.linalg.norm(np.array(feats) - cur_vec, axis=1)
        k       = min(self.n_neighbors, len(valid))
        indices = np.argsort(dists)[:k]
        return [valid[i] for i in indices]

    # ------------------------------------------------------------------
    # GTO action decision
    # ------------------------------------------------------------------

    def decide_action(self, current_song):
        """
        Choose the GTO action (CALL / HOLD / RAISE) whose expected energy
        output is closest to the target on the pre-planned energy arc.

        Expected energies per action
        ----------------------------
        CALL  — current song's energy (sustain)
        HOLD  — midpoint between current and high-cluster median (moderate step)
        RAISE — median energy of the high-energy cluster (big spike)
        """
        if not len(self.energy_arc):
            return self.GTO_RAISE

        step   = min(self.mix_step, len(self.energy_arc) - 1)
        target = self.energy_arc[step]

        cur_energy = self.song_raw_features.get(
            current_song.title, [170.0, 0.0, 0.0]
        )[1]

        high_titles  = [t for t, c in self.song_clusters.items() if c == 2]
        high_vals    = [self.song_raw_features[t][1]
                        for t in high_titles if t in self.song_raw_features]
        raise_energy = float(np.median(high_vals)) if high_vals else cur_energy + 3.0
        hold_energy  = (cur_energy + raise_energy) / 2.0

        action = int(np.argmin([
            abs(cur_energy   - target),  # CALL
            abs(hold_energy  - target),  # HOLD
            abs(raise_energy - target),  # RAISE
        ]))

        logger.debug('GTO step %d: target=%.2f cur=%.2f → %s',
                     step, target, cur_energy, self._ACTION_NAMES[action])
        return action

    # ------------------------------------------------------------------
    # Candidate ranking
    # ------------------------------------------------------------------

    def rank_candidates(self, current_song, candidate_songs, action):
        """
        Return KNN candidates sorted by GTO action preference.

        Preferred candidates (matching the action criterion) come first;
        the remainder are appended as fallbacks so the caller always has
        the full set of options available for transition-quality evaluation.

        Key lookups use the cache built during fit() so no songs need to
        be opened or closed here — important because the caller may have
        master_song open and we must not disturb that state.
        """
        knn = self.get_knn_candidates(current_song, candidate_songs)
        if not knn:
            return list(candidate_songs)[:self.n_neighbors]

        cur_cluster = self.get_cluster(current_song)
        preferred, fallback = [], []

        if action == self.GTO_CALL:
            # Sustain: prefer songs in the same energy cluster.
            for s in knn:
                (preferred if self.get_cluster(s) == cur_cluster else fallback).append(s)

        elif action == self.GTO_HOLD:
            # Harmonic bridge: prefer songs within 1 Camelot step.
            # Use cached keys — avoids opening/closing songs mid-loop.
            ck_pair = self.song_keys.get(current_song.title)
            if ck_pair is None:
                # Fallback: read from live attribute if song is currently open.
                ck = getattr(current_song, 'key', None)
                cs = getattr(current_song, 'scale', None)
            else:
                ck, cs = ck_pair
                # Prefer live attribute if available (song is open).
                ck = getattr(current_song, 'key', None) or ck
                cs = getattr(current_song, 'scale', None) or cs

            if ck is None:
                # Can't determine harmonic compatibility; treat all as fallback.
                return knn

            harmonic = []
            for s in knn:
                sk_pair = self.song_keys.get(s.title)
                if sk_pair is None:
                    fallback.append(s)
                    continue
                sk, ss = sk_pair
                d = camelot_distance(ck, cs, sk, ss)
                if d <= CAMELOT_MAX_DISTANCE:
                    harmonic.append((s, d))
                else:
                    fallback.append(s)
            harmonic.sort(key=lambda x: x[1])
            preferred = [s for s, _ in harmonic]

        elif action == self.GTO_RAISE:
            # Peak drop: prefer songs in the highest-energy cluster.
            for s in knn:
                (preferred if self.get_cluster(s) == 2 else fallback).append(s)

        return preferred + fallback

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def advance(self):
        """Advance the mix-step counter after each successful track transition."""
        self.mix_step += 1
