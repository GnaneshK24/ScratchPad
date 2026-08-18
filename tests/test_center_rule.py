from src.localization.center_rule import resolve_equivalent_peak


def test_center_rule_only_breaks_equivalent_scores_by_search_centre():
    peaks = [(100, 100, .80), (440, 440, .795), (460, 460, .77)]

    equivalent, chosen = resolve_equivalent_peak(peaks, (500., 500.), tolerance=.01)

    assert len(equivalent) == 2
    assert (chosen['x'], chosen['y']) == (490., 490.)

