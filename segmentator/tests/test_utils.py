"""Test utility functions."""

import numpy as np
from tetrahydra.utils import truncate_range, scale_range


def test_truncate_range():
    """Test range truncation."""
    # Given
    data = np.random.random(100)
    data.ravel()[np.random.choice(data.size, 10, replace=False)] = 0
    data.ravel()[np.random.choice(data.size, 5, replace=False)] = np.nan
    p_min, p_max = 2.5, 97.5
    expected = np.nanpercentile(data, [p_min, p_max])
    # When
    output = truncate_range(data, percMin=p_min, percMax=p_max,
                            discard_zeros=False)
    # Then
    assert all(np.nanpercentile(output, [0, 100]) == expected)


def test_scale_range():
    """Test range scaling."""
    # Given
    data = np.random.random(100) - 0.5
    data.ravel()[np.random.choice(data.size, 10, replace=False)] = 0.
    data.ravel()[np.random.choice(data.size, 5, replace=False)] = np.nan
    s = 42.  # scaling factor
    expected = [0., s]  # min and max
    # When
    output = scale_range(data, scale_factor=s, delta=0.01, discard_zeros=False)
    # Then
    assert all([np.nanmin(output) >= expected[0],
                np.nanmax(output) < expected[1]])
