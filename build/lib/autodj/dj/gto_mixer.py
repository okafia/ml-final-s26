import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger('colorlogger')

# Camelot Wheel number for each key, split by scale
_CAMELOT_MAJOR = {
    'B': 1, 'F#': 2, 'Gb': 2, 'C#': 3, 'Db': 3, 'G#': 4, 'Ab': 4,
    'D#': 5, 'Eb': 5, 'A#': 6, 'Bb': 6, 'F': 7, 'C': 8, 'G': 9,
    'D': 10, 'A': 11, 'E': 12,
}
_CAMELOT_MINOR = {
    'G#': 1, 'Ab': 1, 'D#': 2, 'Eb': 2, 'A#': 3, 'Bb': 3, 'F': 4,
    'C': 5, 'G': 6, 'D': 7, 'A': 8, 'E': 9, 'B': 10, 'F#': 11, 'Gb': 11,
    'C#': 12, 'Db': 12,
}


def get_camelot_position(key, scale):
    """Return (number 1-12, mode 'A'=minor / 'B'=major) for a key/scale pair."""
    if scale == 'major':
        return (_CAMELOT_MAJOR.get(key, 8), 'B')
    return (_CAMELOT_MINOR.get(key, 5), 'A')


def camelot_distance(pos1, pos2):
    """
    Distance between two Camelot positions (num, mode).
    0 = same key, 1 = adjacent number on wheel or relative (same number, different mode).
    """
    n1, m1 = pos1
    n2, m2 = pos2
    num_dist = min(abs(n1 - n2), 12 - abs(n1 - n2))
    return num_dist + (0 if m1 == m2 else 1)


def _build_feature_vector(song):
    """
    Extract a 6-dimensional feature vector from an already-open Song.
    Returns (np.ndarray shape (6,), camelot_position tuple).

    Features:
      0 - tempo (BPM)
      1 - energy_ratio: fraction of structural segments labelled 'H'
      2 - sin of Camelot number angle (circular encoding)
      3 - cos of Camelot number angle
      4 - mode: 1.0 = major (B), 0.0 = minor (A)
      5 - theme_val: first component of song_theme_descriptor PCA vector
    """
    bpm = song.tempo if song.tempo is not None else 174.0

    seg_types = song.segment_types
    if seg_types:
        energy_ratio = sum(1 for t in seg_types if t == 'H') / len(seg_types)
    else:
        energy_ratio = 0.5

    key = song.key if song.key else 'C'
    scale = song.scale if song.scale else 'major'
    camelot_pos = get_camelot_position(key, scale)
    angle = 2 * np.pi * (camelot_pos[0] - 1) / 12
    mode_val = 1.0 if camelot_pos[1] == 'B' else 0.0

    theme = getattr(song, 'song_theme_descriptor', None)
    theme_val = float(theme[0]) if (theme is not None and len(theme) > 0) else 0.0

    return np.array([bpm, energy_ratio, np.sin(angle), np.cos(angle), mode_val, theme_val]), camelot_pos


class GTOMixer:
    """
    GTO-inspired track selection using K-Means clustering (k=3) and KNN retrieval.

    Clusters map to three GTO actions driven by crowd energy input:
      crowd 'low'    -> action 'raise'  : select from highest-energy cluster
      crowd 'medium' -> action 'hold'   : select harmonically nearest (Camelot ≤ 1)
      crowd 'high'   -> action 'call'   : select from same cluster (sustain energy)

    Call fit() once after the song collection is ready, then use apply_gto_filter()
    inside the track selection pipeline to re-rank candidate songs.
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.fitted = False
        self._songs = []
        self._titles = []
        self._features_raw = None
        self._features_scaled = None
        self._camelot = {}             # title -> (num, mode)
        self._cluster_id = {}          # title -> int
        self._cluster_energy_label = {}  # cluster int -> 'low'/'mid'/'high'
        self._knn = None
        self._scaler = StandardScaler()
        self._kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    # ------------------------------------------------------------------
    def fit(self, songs):
        """
        Fit K-Means and KNN on the annotated song collection.
        Each song is opened, features are extracted, then the song is closed again.
        """
        valid = []
        for s in songs:
            s.open()
            if s.tempo is not None:
                valid.append(s)
            else:
                s.close()

        if len(valid) < 3:
            for s in valid:
                s.close()
            logger.warning('GTOMixer: need at least 3 annotated songs, skipping fit.')
            return

        self._songs = valid
        self._titles = [s.title for s in valid]

        raw = []
        for s in valid:
            feat, cpos = _build_feature_vector(s)
            raw.append(feat)
            self._camelot[s.title] = cpos
            s.close()

        self._features_raw = np.array(raw)
        self._features_scaled = self._scaler.fit_transform(self._features_raw)

        labels = self._kmeans.fit_predict(self._features_scaled)

        # Label clusters low/mid/high by median energy ratio (feature index 1)
        medians = {
            c: np.median(self._features_raw[labels == c, 1])
            for c in range(3) if np.any(labels == c)
        }
        for rank_idx, cid in enumerate(sorted(medians, key=medians.get)):
            self._cluster_energy_label[cid] = ('low', 'mid', 'high')[rank_idx]

        for s, cid in zip(valid, labels):
            self._cluster_id[s.title] = int(cid)

        k = min(self.n_neighbors + 1, len(valid))
        self._knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        self._knn.fit(self._features_scaled)

        self.fitted = True
        logger.debug('GTOMixer fitted on {} tracks. Cluster energy labels: {}'.format(
            len(valid), self._cluster_energy_label))

    # ------------------------------------------------------------------
    @staticmethod
    def decide_action(crowd_energy):
        """
        Map crowd energy level to a GTO action string.
          'low'    -> 'raise'  (energize the crowd with a high-energy track)
          'medium' -> 'hold'   (smooth harmonic bridge to maintain flow)
          'high'   -> 'call'   (sustain the current energy level)
        """
        return {'low': 'raise', 'medium': 'hold', 'high': 'call'}.get(crowd_energy, 'hold')

    # ------------------------------------------------------------------
    def apply_gto_filter(self, song_options, current_song, crowd_energy):
        """
        Re-rank song_options according to the GTO action for the current crowd
        energy level.  Higher-priority candidates are placed first so the
        downstream theme/ODF selection still refines within the preferred pool.

        Parameters
        ----------
        song_options  : np.ndarray of Song objects (from getSongOptionsInKey)
        current_song  : Song  (currently playing, must be open)
        crowd_energy  : str   'low' | 'medium' | 'high'

        Returns
        -------
        np.ndarray of Songs, re-ordered by GTO priority.
        """
        if not self.fitted or len(song_options) == 0:
            return song_options

        action = self.decide_action(crowd_energy)
        current_cluster = self._cluster_id.get(current_song.title)
        current_camelot = self._camelot.get(current_song.title, (8, 'B'))

        if action == 'call':
            # Sustain: prefer songs in the same energy cluster
            preferred = [s for s in song_options
                         if self._cluster_id.get(s.title) == current_cluster]
            rest = [s for s in song_options
                    if self._cluster_id.get(s.title) != current_cluster]
            ordered = preferred + rest

        elif action == 'hold':
            # Harmonic bridge: sort by ascending Camelot Wheel distance
            ordered = sorted(
                song_options,
                key=lambda s: camelot_distance(
                    current_camelot,
                    self._camelot.get(s.title, (8, 'B'))
                )
            )

        else:  # 'raise'
            # Peak drop: prefer tracks from the highest-energy cluster
            high_cid = next(
                (c for c, lbl in self._cluster_energy_label.items() if lbl == 'high'),
                None
            )
            preferred = [s for s in song_options
                         if self._cluster_id.get(s.title) == high_cid]
            rest = [s for s in song_options
                    if self._cluster_id.get(s.title) != high_cid]
            ordered = preferred + rest

        logger.debug(
            'GTO action={} (crowd={}) for "{}": {} candidates, top choice "{}".'.format(
                action, crowd_energy, current_song.title[:24],
                len(ordered),
                ordered[0].title[:24] if ordered else 'none'
            )
        )
        return np.array(ordered)

    # ------------------------------------------------------------------
    def get_energy_label(self, song):
        """Return the energy label ('low', 'mid', 'high') assigned to a song."""
        cid = self._cluster_id.get(song.title)
        if cid is None:
            return 'mid'
        return self._cluster_energy_label.get(cid, 'mid')
