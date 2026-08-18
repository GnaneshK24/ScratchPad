"""Classical coarse-to-fine SEM localization; no learned features or training."""
from __future__ import annotations
from dataclasses import dataclass
import time
import cv2
import numpy as np
from .coordinates import center_to_bbox, top_left_to_center

@dataclass(frozen=True)
class ClassicalMatcherConfig:
    intensity_weight: float = .12
    contrast_weight: float = .10
    gradient_weight: float = .18
    gx_weight: float = .10
    gy_weight: float = .10
    edge_weight: float = .10
    local_structure_weight: float = .12
    bandpass_weight: float = .10
    shape_weight: float = .08
    top_k: int = 100
    cheap_rerank_top_k: int = 30
    verify_top_k: int = 30
    coarse_peaks_per_representation: int = 64
    use_distinctiveness: bool = False
    pyramid_levels: int = 3
    rotation_range: tuple[float, float] = (-3., 3.)
    rotation_step: float = 1.5
    scales: tuple[float, ...] = (.95, .975, 1., 1.025, 1.05)
    refinement_window: int = 70
    nms_radius: int = 20
    clahe_clip_limit: float = 2.
    ecc_iterations: int = 50
    ecc_epsilon: float = 1e-5
    equivalent_score_tolerance: float = .012
    use_center_tie_break: bool = True
    def rotations(self):
        return tuple(float(v) for v in np.arange(self.rotation_range[0], self.rotation_range[1] + self.rotation_step / 2, self.rotation_step))

def _gray(image):
    if image is None: raise ValueError('Image could not be loaded')
    if image.ndim == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2: raise ValueError('Images must be grayscale or BGR arrays')
    return image

def _unit(image):
    image = image.astype(np.float32, copy=False); lo, hi = np.percentile(image, (1, 99))
    # Sparse templates can occupy <=1% of a search image; retain their signal.
    if hi <= lo + 1e-6:
        lo, hi = float(image.min()), float(image.max())
    return np.zeros_like(image, dtype=np.float32) if hi <= lo + 1e-6 else np.clip((image - lo) / (hi - lo), 0, 1).astype(np.float32)

def _representations(image, config):
    """Robust intensity, directional structure, and geometry maps."""
    intensity = _unit(_gray(image)); u8 = np.uint8(np.round(intensity * 255))
    local = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255
    gx = cv2.Sobel(local, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(local, cv2.CV_32F, 0, 1, ksize=3)
    gradient = _unit(cv2.magnitude(gx, gy)); edges = cv2.Canny(u8, 40, 120).astype(np.float32) / 255
    edge = _unit(cv2.GaussianBlur(edges, (0, 0), 1.2) + .35 * gradient)
    distance = _unit(cv2.distanceTransform((edges < .5).astype(np.uint8), cv2.DIST_L2, 3))
    structure = _unit(np.abs(cv2.Laplacian(local, cv2.CV_32F, ksize=3)))
    bandpass = _unit(local - cv2.GaussianBlur(local, (0, 0), 2.5))
    return {'intensity': intensity, 'local': local, 'gradient': gradient,
            'gx': _unit(gx), 'gy': _unit(gy), 'bandpass': bandpass,
            'edge': edge, 'distance': distance, 'structure': structure}

def _variant(image, angle, scale):
    h, w = image.shape; matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

def _distinctiveness(image):
    """Reference-only reliability map for weighted ZNCC.

    Long periodic lines carry little localization information.  Corners,
    line-endings, direction changes and locally unusual texture receive more
    weight.  The map is derived entirely from the input reference.
    """
    image = image.astype(np.float32, copy=False)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = _unit(cv2.magnitude(gx, gy))
    harris = _unit(cv2.cornerHarris(image, 2, 3, .04))
    mean = cv2.GaussianBlur(image, (0, 0), 2.0)
    variance = _unit(cv2.GaussianBlur((image - mean) ** 2, (0, 0), 1.5))
    reliability = _unit(.35 * magnitude + .45 * harris + .20 * variance)
    # Preserve enough support for a well-conditioned local normalization.
    floor = float(np.percentile(reliability, 55))
    return np.clip((reliability - floor) / max(1. - floor, 1e-6), 0, 1).astype(np.float32) + .05

def _weighted_zncc(search, reference, weights):
    """Fast weighted zero-mean normalized cross-correlation for one template."""
    weights = weights.astype(np.float32, copy=False)
    total = float(weights.sum())
    ref_mean = float((weights * reference).sum() / max(total, 1e-6))
    centered_ref = weights * (reference - ref_mean)
    ref_energy = float((weights * (reference - ref_mean) ** 2).sum())
    numerator = cv2.matchTemplate(search, centered_ref, cv2.TM_CCORR)
    search_sum = cv2.matchTemplate(search, weights, cv2.TM_CCORR)
    search_square_sum = cv2.matchTemplate(search * search, weights, cv2.TM_CCORR)
    search_energy = np.maximum(search_square_sum - search_sum * search_sum / max(total, 1e-6), 0.)
    return numerator / np.sqrt(np.maximum(search_energy * max(ref_energy, 1e-12), 1e-12))

class ClassicalSEMLocalizer:
    """Multi-representation, rotation-aware, coarse-to-fine SEM localizer."""
    def __init__(self, config=None, top_k=None, rotations=None, scales=None):
        config = config or ClassicalMatcherConfig()
        if top_k is not None or rotations is not None or scales is not None:
            values = dict(config.__dict__)
            if top_k is not None: values['top_k'] = top_k
            if rotations is not None:
                values['rotation_range'] = (min(rotations), max(rotations))
                if len(rotations) > 1: values['rotation_step'] = rotations[1] - rotations[0]
            if scales is not None: values['scales'] = tuple(scales)
            config = ClassicalMatcherConfig(**values)
        self.config = config

    def _fused(self, search, reference):
        c = self.config
        return (c.intensity_weight * cv2.matchTemplate(search['intensity'], reference['intensity'], cv2.TM_CCOEFF_NORMED)
              + c.contrast_weight * cv2.matchTemplate(search['local'], reference['local'], cv2.TM_CCOEFF_NORMED)
              + c.gradient_weight * cv2.matchTemplate(search['gradient'], reference['gradient'], cv2.TM_CCOEFF_NORMED)
              + c.gx_weight * cv2.matchTemplate(search['gx'], reference['gx'], cv2.TM_CCOEFF_NORMED)
              + c.gy_weight * cv2.matchTemplate(search['gy'], reference['gy'], cv2.TM_CCOEFF_NORMED)
              + c.edge_weight * cv2.matchTemplate(search['edge'], reference['edge'], cv2.TM_CCOEFF_NORMED)
              + c.shape_weight * cv2.matchTemplate(search['distance'], reference['distance'], cv2.TM_CCOEFF_NORMED)
              + c.local_structure_weight * cv2.matchTemplate(search['structure'], reference['structure'], cv2.TM_CCOEFF_NORMED)
              + c.bandpass_weight * cv2.matchTemplate(search['bandpass'], reference['bandpass'], cv2.TM_CCOEFF_NORMED))

    @staticmethod
    def _peaks(score_map, count, radius):
        work = score_map.copy(); result = []
        for _ in range(min(count, work.size)):
            _, score, _, (x, y) = cv2.minMaxLoc(work)
            if not np.isfinite(score): break
            result.append((x, y, float(score)))
            cv2.rectangle(work, (max(0, x-radius), max(0, y-radius)), (min(work.shape[1]-1, x+radius), min(work.shape[0]-1, y+radius)), -np.inf, -1)
        return result

    def _coarse(self, search, reference, limit=None, trace=None):
        """Build candidates independently, then combine with spatial NMS.

        A fused response map can discard a true peak when one otherwise-useful
        representation is corrupted by SEM noise.  Each representation gets a
        separate candidate budget; only the expensive local verification fuses
        their evidence.
        """
        factor = 2 ** (self.config.pyramid_levels - 1)
        s = cv2.resize(search, (search.shape[1] // factor, search.shape[0] // factor), interpolation=cv2.INTER_AREA)
        search_repr = _representations(s, self.config)
        sources = ('intensity', 'local', 'gradient', 'gx', 'gy', 'bandpass', 'edge', 'structure')
        if self.config.use_distinctiveness:
            sources += ('distinctive',)
        best_scores = {name: None for name in sources}
        best_angles = {name: None for name in sources}
        for angle in self.config.rotations():
            rotated = _variant(reference, angle, 1.) if angle else reference
            r = cv2.resize(rotated, (rotated.shape[1] // factor, rotated.shape[0] // factor), interpolation=cv2.INTER_AREA)
            reference_repr = _representations(r, self.config)
            reliability = _distinctiveness(reference_repr['local'])
            for name in sources:
                scores = (_weighted_zncc(search_repr['local'], reference_repr['local'], reliability)
                          if name == 'distinctive' else
                          cv2.matchTemplate(search_repr[name], reference_repr[name], cv2.TM_CCOEFF_NORMED))
                if best_scores[name] is None:
                    best_scores[name] = scores
                    best_angles[name] = np.full(scores.shape, angle, dtype=np.float32)
                else:
                    replace = scores > best_scores[name]
                    best_scores[name][replace] = scores[replace]
                    best_angles[name][replace] = angle

        raw = []
        radius = max(1, self.config.nms_radius // factor)
        for name in sources:
            scores = best_scores[name]
            # Scores from distinct representations are not directly comparable.
            # Use each map's robust peak prominence only to prioritize NMS.
            median, p95 = np.percentile(scores, (50, 95))
            spread = max(float(p95 - median), 1e-6)
            for rank, (x, y, score) in enumerate(self._peaks(scores, self.config.coarse_peaks_per_representation, radius)):
                raw.append({'x': x * factor, 'y': y * factor, 'coarse_score': score,
                            'coarse_rotation': float(best_angles[name][y, x]),
                            'candidate_source': name,
                            'candidate_priority': float((score - median) / spread - rank * 1e-3)})
        raw.sort(key=lambda item: item['candidate_priority'], reverse=True)
        if trace is not None:
            trace['candidate_union_raw'] = [dict(item) for item in raw]
        selected, candidate_limit = [], limit or self.config.top_k
        for item in raw:
            if all(np.hypot(item['x'] - other['x'], item['y'] - other['y']) >= self.config.nms_radius for other in selected):
                selected.append(item)
                if len(selected) >= candidate_limit:
                    break
        if trace is not None:
            trace['rank_after_nms'] = [dict(item) for item in selected]
        return selected

    def candidate_diagnostics(self, search_image, reference_image, limit=100):
        """Return spatially distinct raw candidates for recall benchmarking.

        Each representation is measured independently at full resolution.  This
        is diagnostic/selection work; expensive rotation and ECC remain local.
        """
        search, reference = _gray(search_image), _gray(reference_image)
        if search.shape != (1000, 1000): raise ValueError(f'search must be 1000x1000, got {search.shape}')
        if reference.shape == (1000, 1000): reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
        if reference.shape != (100, 100): raise ValueError(f'reference must be 100x100, got {reference.shape}')
        s, r = _representations(search, self.config), _representations(reference, self.config)
        result = {}
        for name in s:
            score_map = cv2.matchTemplate(s[name], r[name], cv2.TM_CCOEFF_NORMED)
            result[name] = [{'center_x': float(x + 50), 'center_y': float(y + 50), 'score': score}
                            for x, y, score in self._peaks(score_map, limit, self.config.nms_radius)]
        fused = self._fused(s, r)
        result['fused_full'] = [{'center_x': float(x + 50), 'center_y': float(y + 50), 'score': score}
                                for x, y, score in self._peaks(fused, limit, self.config.nms_radius)]
        coarse = self._coarse(search, reference, limit=limit)
        result['independent_coarse'] = [{'center_x': float(item['x'] + 50), 'center_y': float(item['y'] + 50),
                                         'score': item['coarse_score'], 'source': item['candidate_source']}
                                        for item in coarse]
        # Backwards-compatible key consumed by the existing benchmark report.
        result['fused_coarse'] = result['independent_coarse']
        return result
    def _verify(self, search, reference, candidate):
        h, w = reference.shape; m = self.config.refinement_window; ex, ey = round(candidate['x']), round(candidate['y'])
        x0, y0 = max(0, ex-m), max(0, ey-m); x1, y1 = min(1000, ex+w+m), min(1000, ey+h+m)
        if x1-x0 < w or y1-y0 < h: return None
        local = {key: value[y0:y1, x0:x1] for key, value in search.items()}; best = None
        coarse_rotation = float(candidate.get('coarse_rotation', 0.))
        angles = sorted(self.config.rotations(), key=lambda value: abs(value - coarse_rotation))[:3]
        for angle in angles:
            for scale in self.config.scales:
                ref = _representations(_variant(reference, angle, scale), self.config); score_map = self._fused(local, ref)
                _, score, _, (x, y) = cv2.minMaxLoc(score_map)
                components = {key: float(cv2.matchTemplate(local[key], ref[key], cv2.TM_CCOEFF_NORMED)[y, x]) for key in local}
                item = {'x': float(x0+x), 'y': float(y0+y), 'score': float(score), 'rotation': float(angle), 'scale': float(scale), 'components': components, 'coarse_score': candidate['coarse_score'], 'candidate_source': candidate.get('candidate_source')}
                if best is None or item['score'] > best['score']: best = item
        return best

    def _cheap_rerank(self, search, reference, candidates):
        """Use full-resolution structural evidence before costly variants.

        This stage evaluates a small translation neighbourhood at the candidate's
        coarse rotation, but does not run a scale sweep, phase correlation, or
        ECC.  It permits broad candidate recall without multiplying expensive
        refinement cost by the candidate count.
        """
        margin = 24
        ranked = []
        for candidate in candidates:
            ex, ey = round(candidate['x']), round(candidate['y'])
            x0, y0 = max(0, ex - margin), max(0, ey - margin)
            x1, y1 = min(1000, ex + 100 + margin), min(1000, ey + 100 + margin)
            if x1 - x0 < 100 or y1 - y0 < 100:
                continue
            angle = float(candidate.get('coarse_rotation', 0.))
            transformed = _variant(reference, angle, 1.) if angle else reference
            local = {name: image[y0:y1, x0:x1] for name, image in search.items()}
            score_map = self._fused(local, _representations(transformed, self.config))
            _, score, _, (x, y) = cv2.minMaxLoc(score_map)
            ranked.append({**candidate, 'x': float(x0 + x), 'y': float(y0 + y),
                           'cheap_score': float(score)})
        ranked.sort(key=lambda item: item['cheap_score'], reverse=True)
        return ranked

    def _align(self, patch, reference):
        phase_shift, phase = cv2.phaseCorrelate(reference.astype(np.float32), patch.astype(np.float32)); warp = np.eye(2, 3, dtype=np.float32)
        try:
            ecc, warp = cv2.findTransformECC(reference.astype(np.float32), patch.astype(np.float32), warp, cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.config.ecc_iterations, self.config.ecc_epsilon))
            dx, dy = -float(warp[0,2]), -float(warp[1,2])
            if abs(dx) <= 4 and abs(dy) <= 4: return dx, dy, float(np.clip(ecc, -1, 1)), float(np.clip(phase, 0, 1))
        except cv2.error: pass
        return 0., 0., 0., float(np.clip(phase, 0, 1))

    def localize(self, search_image, reference_image, *, rotations=None, scales=None):
        if rotations is not None or scales is not None:
            return ClassicalSEMLocalizer(self.config, rotations=rotations, scales=scales).localize(search_image, reference_image)
        started = time.perf_counter(); search, reference = _gray(search_image), _gray(reference_image)
        if search.shape != (1000, 1000): raise ValueError(f'search must be 1000x1000, got {search.shape}')
        if reference.shape == (1000, 1000): reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
        if reference.shape != (100, 100): raise ValueError(f'reference must be 100x100, got {reference.shape}')
        search_repr = _representations(search, self.config)
        coarse = self._coarse(search, reference, limit=self.config.top_k)
        cheap = self._cheap_rerank(search_repr, reference, coarse)[:self.config.cheap_rerank_top_k]
        candidates = [item for c in cheap[:self.config.verify_top_k] if (item := self._verify(search_repr, reference, c)) is not None]
        if not candidates: raise RuntimeError('No valid localization candidate')
        candidates.sort(key=lambda item: item['score'], reverse=True)
        # Collapse overlapping local refinements before ambiguity/confidence checks.
        distinct = []
        for candidate in candidates:
            if all(np.hypot(candidate['x']-other['x'], candidate['y']-other['y']) >= 100 for other in distinct): distinct.append(candidate)
        candidates = distinct or candidates[:1]
        # The challenge's centre rule is a tie-breaker for genuinely equivalent
        # periodic matches, never a global spatial prior.
        appearance_best = candidates[0]['score']
        equivalent = [item for item in candidates if appearance_best - item['score'] <= self.config.equivalent_score_tolerance]
        best = (min(equivalent, key=lambda item: np.hypot(item['x'] + 50 - 500, item['y'] + 50 - 500))
                if self.config.use_center_tie_break else candidates[0])
        candidates.remove(best)
        candidates.insert(0, best)
        second = candidates[1] if len(candidates) > 1 else None
        x, y = round(best['x']), round(best['y']); ref_local = _representations(_variant(reference, best['rotation'], best['scale']), self.config)['local']
        dx, dy, ecc, phase = self._align(search_repr['local'][y:y+100, x:x+100], ref_local); best['x'] += dx; best['y'] += dy
        cx, cy = top_left_to_center(best['x'], best['y']); second_score = None if second is None else second['score']; margin = best['score'] - (second_score if second_score is not None else -1.)
        ratio = best['score'] / max(abs(second_score) if second_score is not None else 1e-6, 1e-6); disagreement = float(np.std(list(best['components'].values())))
        confidence = float(np.clip(max(0., margin) / max(1.-abs(best['score']), 1e-6) * (1.-min(disagreement, 1.)), 0., 1.))
        output_candidates = [{**c, 'center_x': top_left_to_center(c['x'], c['y'])[0], 'center_y': top_left_to_center(c['x'], c['y'])[1]} for c in candidates]
        return {'x': cx, 'y': cy, 'center_x': cx, 'center_y': cy, 'bbox': center_to_bbox(cx, cy), 'confidence': confidence, 'low_confidence': confidence < .05, 'score': best['score'], 'peak_score': best['score'], 'second_score': second_score, 'second_peak_score': second_score, 'peak_margin': margin, 'peak_ratio': ratio, 'rotation': best['rotation'], 'scale': best['scale'], 'ecc_score': ecc, 'phase_response': phase, 'component_scores': best['components'], 'top_k': output_candidates, 'method': 'multirepresentation_coarse_to_fine', 'model': 'classical_multirepresentation', 'inference_time_ms': (time.perf_counter()-started)*1000}

ClassicalSEMMatcher = ClassicalSEMLocalizer







