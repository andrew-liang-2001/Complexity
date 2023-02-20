import numpy as np
import numba as nb
import warnings


L = np.array([4, 8, 16, 32, 64, 128, 256])  # define L in the namespace
L_big = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
t_c = np.array([100, 200, 400, 1500, 5000, 18000, 70000, 220_000, 1_000_000])


class OsloModel:
    """A class for the Oslo Model"""

    def __init__(self, L: int, z_threshold_max: int = 2):
        """
        Initialise the Oslo Model
        :param L: size of the system L
        :param p: probability of z_threshold = 1
        """
        self.z_threshold_max = z_threshold_max + 1
        self.L = L
        self._heights = np.zeros(L, dtype=np.int32)
        self.z_threshold = np.random.randint(1, self.z_threshold_max, size=L, dtype=np.int16)
        self.z = np.zeros(L, dtype=np.int32)
        self.time = 0

    def __str__(self):
        max_height = np.max(self._heights)
        if max_height >= 20 or self.L >= 20:
            return self.__repr__()
        elif max_height == 0:
            return str(np.zeros((1, self.L), dtype=int))
        else:
            return f"The Oslo Model looks like:\n{self.heights_matrix()}"

    def __repr__(self):
        return f"The z values are {self.z}\nThe heights are {self._heights}"

    def heights_matrix(self):
        """
        Return the heights of the model as a matrix of 1's and 0's
        """
        max_height = np.max(self._heights)
        matrix = np.zeros((max_height, self.L), dtype=int)
        for i in range(self.L):
            if self._heights[i] > 0:
                matrix[-self._heights[i]:, i] = 1
        return matrix

    @property
    def heights(self):
        """
        Get the heights of the model
        :return: heights
        """
        return self._heights

    @heights.setter
    def heights(self, value) -> None:
        """
        Set the heights of the model. This runs a series of checks to verify that the heights configurations
        are valid. Also calculate the corresponding z vector

        :param value: new heights configuration
        """
        if value.dtype != int:
            raise TypeError("Heights must contain integers only")
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
        """
        Run the model for a single iteration by placing a grain on the leftmost site. Return the avalanche size s.
        A wrapper for the numba-compiled function _run
        :return: avalanche size s
        """

        self._heights, self.z, self.z_threshold, self.time, s = _run(self._heights, self.z, self.z_threshold, self.L,
                                                                     self.time, self.z_threshold_max)
        return s

    def run_z(self) -> int:
        """
        Run the model for a single iteration by placing a grain on the leftmost site. Return the avalanche size s.
        A wrapper for the numba-compiled function _run. This version only updates the slopes.
        """

        self.z, self.z_threshold, self.time, diff = _run_z(self.z, self.z_threshold, self.L,
                                                                     self.time, self.z_threshold_max)
        return diff

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


@nb.njit
def _run(heights, z, z_threshold, L: int, time: int, z_threshold_max: int):
    """
    Run the model for a single iteration by placing a grain on the leftmost site. Return the avalanche size s.

    :param heights: array of heights of size L
    :param z: array of z of size L
    :param z_threshold: array of z_thresholds of size L
    :param L: length of the model
    :param time: time of the system t
    :return: avalanche size s
    """
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
            z_threshold[0] = np.random.randint(1, z_threshold_max)
            s += 1
        for i in range(1, L - 1):
            if z[i] > z_threshold[i]:
                z[i] -= 2
                z[i + 1] += 1
                z[i - 1] += 1
                heights[i] -= 1
                heights[i + 1] += 1
                z_threshold[i] = np.random.randint(1, z_threshold_max)
                s += 1
        if z[-1] > z_threshold[-1]:
            z[-1] -= 1
            z[-2] += 1
            heights[-1] -= 1
            z_threshold[-1] = np.random.randint(1, z_threshold_max)
            s += 1

    time += 1
    return heights, z, z_threshold, time, s


@nb.njit
def _run_z(z, z_threshold, L: int, time: int, z_threshold_max: int):
    """
    Run the model for a single iteration by placing a grain on the leftmost site. Faster than _run because it does
    not update the heights array. Return the difference in height of the pile.

    :param z: array of z of size L
    :param z_threshold: array of z_thresholds of size L
    :param L: length of the model
    :param time: time of the system t
    :return: avalanche size s
    """
    z[0] += 1
    diff = 1

    while not np.all(z <= z_threshold):  # stability condition also hardcoded for performance
        """This loops brings the system to the next stable state. z and heights are hardcoded independently for
        performance reasons"""
        if z[0] > z_threshold[0]:
            z[0] -= 2
            z[1] += 1
            z_threshold[0] = np.random.randint(1, z_threshold_max)
            diff -= 1
        for i in range(1, L - 1):
            if z[i] > z_threshold[i]:
                z[i] -= 2
                z[i + 1] += 1
                z[i - 1] += 1
                z_threshold[i] = np.random.randint(1, z_threshold_max)
        if z[-1] > z_threshold[-1]:
            z[-1] -= 1
            z[-2] += 1
            z_threshold[-1] = np.random.randint(1, z_threshold_max)

    time += 1
    return z, z_threshold, time, diff


def truncated_series(L, a0, a1, w1):
    return a0 * L * (1-a1 * L ** (-w1))


if __name__ == "__main__":
    model = OsloModel(4)
    for _ in range(10):
        model.run()
        print(model)
