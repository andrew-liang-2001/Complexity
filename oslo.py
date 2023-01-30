import numpy as np
import numba as nb
import warnings

L = np.array([4, 8, 16, 32, 64, 128, 256])  # define L in the namespace


class OsloModel:
    """A class for the Oslo Model"""

    def __init__(self, L: int):
        """Note that under the numba implementation, creation of multiple models is slow because of the need for
        random arrays to be looped over."""
        self.L = L
        self._heights = np.zeros(L, dtype=np.int32)
        self.z_threshold = np.random.choice([1, 2], size=L, p=[0.5, 0.5])
        temp = np.append(self._heights, np.array(0, dtype=np.int32))
        self.z = temp[:-1] - temp[1:]
        self.time = 0

    def __str__(self):
        max_height = np.max(self._heights)
        if max_height >= 20 or self.L >= 20:
            return self.__repr__()
        elif max_height == 0:
            return str(np.zeros((1, self.L), dtype=int))
        else:
            matrix = np.zeros((max_height, self.L), dtype=int)
            for i in range(self.L):
                if self._heights[i] > 0:
                    matrix[-self._heights[i]:, i] = 1
            return f"The Oslo Model looks like:\n{str(matrix)}"

    def __repr__(self):
        return f"The z vector is {self.z}\nThe heights are {self._heights}"

    @property
    def heights(self):
        return self._heights

    @heights.setter
    def heights(self, value):
        """Set the heights of the model. This runs a series of checks to verify that the heights configurations
        are valid. Also calculates the corresponding z vector."""
        if not isinstance(value, np.ndarray):
            raise TypeError("Heights must be a numpy array")
        if value.dtype != int:
            raise TypeError("Heights must be an array of integers")
        if value.shape[0] != self.L:
            raise ValueError("Heights must have the same length as L")
        if np.any(value < 0):
            raise ValueError("Heights must be positive")
        if not self.is_system_stable():
            warnings.warn("Heights configuration supplied is unstable")

        self.heights = value

        temp = np.append(self._heights, np.array(0, dtype=np.int32))
        self.z = temp[:-1] - temp[1:]

    def is_system_stable(self) -> bool:
        """Check if the system is stable by an elementwise comparison of the z and z_threshold vectors.
        It is O(L) and may not always be the most efficient"""
        return np.all(self.z <= self.z_threshold)

    def run(self) -> int:
        """A wrapper for the numba compiled function _run"""
        self._heights, self.z, self.z_threshold, self.time, s = _run(self._heights, self.z, self.z_threshold, self.L,
                                                                     self.time)
        return s

    def reset(self) -> None:
        """Reset the model to its initial state"""
        self.__init__(self.L)

    def unstable_indices(self):
        """Return the indices of the unstable sites"""
        return np.where(self.z > self.z_threshold)[0]

    def doctor(self):
        """Perform a few standard checks of common issues within the instance of OsloModel"""
        temp2 = np.append(self._heights, np.array(0, dtype=np.int32))
        if not self.is_system_stable():
            print("Current configuration is not stable")
        elif not np.all(self.z, -np.diff(temp2)):
            print("z and heights are not consistent. Check the run method")
        else:
            "No issues found"


@nb.njit(
def _run(heights, z, z_threshold, L: int, time: int):
    """Run the model for a single iteration by placing a grain on the leftmost site. Return the avalanche size s."""
    heights[0] += 1
    z[0] += 1
    s = 0

    while not np.all(z <= z_threshold):  # stability condition also hardcoded for performance
        """This loops brings the system to the next stable state. z and heights are hardcoded independently for
        performance reasons"""
        if z[0] > z_threshold[0]:
            z[0] -= 2
            z[1] += 1
            heights[0] -= 1
            heights[1] += 1
            z_threshold[0] = np.random.randint(1, 3)
            s += 1
        for i in range(1, L - 1):
            if z[i] > z_threshold[i]:
                z[i] -= 2
                z[i + 1] += 1
                z[i - 1] += 1
                heights[i] -= 1
                heights[i + 1] += 1
                z_threshold[i] = np.random.randint(1, 3)
                s += 1
        if z[-1] > z_threshold[-1]:
            z[-1] -= 1
            z[-2] += 1
            heights[-1] -= 1
            z_threshold[-1] = np.random.randint(1, 3)
            s += 1

    time += 1
    return heights, z, z_threshold, time, s


if __name__ == "__main__":
    model = OsloModel(16)
    model.run()
