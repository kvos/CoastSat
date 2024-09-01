import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from coastsat import SDS_tools

# Test 1: Verify that NumPy is working correctly
def test_numpy():
    array = np.array([1, 2, 3])
    assert np.sum(array) == 6, "NumPy sum test failed!"
    print("Test NumPy: OK")

# Test 2: Verify that Matplotlib is working and can create a simple plot
def test_matplotlib():
    plt.figure()
    plt.plot([0, 1, 2], [0, 1, 4])
    plt.title("Test Plot")
    plt.savefig("test_plot.png")
    assert os.path.exists("test_plot.png"), "Matplotlib plot test failed!"
    print("Test Matplotlib: OK")

# Test 3: Verify that Pandas is working correctly
def test_pandas():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert df.shape == (3, 2), "Pandas DataFrame test failed!"
    print("Test Pandas: OK")

# Test 4: Verify that basic CoastSat functions are working
def test_coastsat():
    polygon = [[[151.301454, -33.700754],
                [151.311453, -33.702075],
                [151.307237, -33.739761],
                [151.294220, -33.736329],
                [151.301454, -33.700754]]]
    
    # Test the smallest_rectangle function
    rectangle = SDS_tools.smallest_rectangle(polygon)
    
    # Print the output for debugging
    print("Output of smallest_rectangle:", rectangle)
    
    # Check that the output is a list of exactly 4 vertices
    assert len(rectangle[0]) == 5, "CoastSat smallest_rectangle test failed! The rectangle should have 5 points (4 vertices plus the closing point)."
    assert rectangle[0][0] == rectangle[0][-1], "The first and last point should be the same to close the polygon."
    
    print("Test CoastSat: OK")

if __name__ == "__main__":
    test_numpy()
    test_matplotlib()
    test_pandas()
    test_coastsat()

    print("All tests were successfully executed.")
