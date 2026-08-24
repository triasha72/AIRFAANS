import numpy as np

from airfaans.normalization import Normalization, RunningMoments


def test_running_moments_matches_direct_statistics():
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    moments = RunningMoments(3)
    moments.update(values[:4])
    moments.update(values[4:])
    mean, scale = moments.finish()
    np.testing.assert_allclose(mean, values.mean(axis=0))
    np.testing.assert_allclose(scale, values.std(axis=0), rtol=1e-6)


def test_normalization_round_trip_and_serialization():
    normalization = Normalization(
        feature_mean=np.array([1.0, 2.0], dtype=np.float32),
        feature_scale=np.array([2.0, 4.0], dtype=np.float32),
        target_mean=np.arange(4, dtype=np.float32),
        target_scale=np.arange(1, 5, dtype=np.float32),
    )
    targets = np.arange(12, dtype=np.float32).reshape(3, 4)
    restored = Normalization.from_dict(normalization.to_dict())
    np.testing.assert_allclose(
        restored.inverse_targets(restored.transform_targets(targets)), targets
    )
