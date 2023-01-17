import numpy as np
from tqdm import tqdm
import warnings
import cProfile
from numba.experimental import jitclass
from numba import int32
"""Numba is so fast. All hail numba"""


@jitclass(spec=[
    ('L', int32),
    ('_heights', int32[:]),
    ('time', int32),  # largest int32 is 2,147,483,647 so time may be an issue if the model is run for too long
    ("z_threshold", int32[:]),
    ("z", int32[:]),
])
class OsloModel:
    """A class for the Oslo Model"""
    def __init__(self, L: int):
        self.L = L
        self._heights = np.zeros(L, dtype=np.int32)
        temp = np.zeros(L, dtype=np.int32)
        for i in range(L):
            temp[i] = np.random.randint(1, 3)
        self.z_threshold = temp
        temp2 = np.append(self._heights, np.array(0, dtype=np.int32))
        self.z = temp2[:-1] - temp2[1:]
        self.time = 0

    # @property
    # def heights(self):
    #     return self._heights
    #
    # @heights.setter
    # def heights(self, value):
    #     if not isinstance(value, np.ndarray):
    #         raise TypeError("Heights must be a numpy array")
    #     if value.dtype != int:
    #         raise TypeError("Heights must be an array of integers")
    #     if value.ndim != 1:
    #         raise ValueError("Heights must be a 1D array")
    #     if value.shape[0] != self.L:
    #         raise ValueError("Heights must have the same length as L")
    #     self.heights = value
    #
    #     temp2 = np.append(self._heights, np.array(0, dtype=np.int32))
    #     self.z = temp2[:-1] - temp2[1:]

    # def __str__(self):
    #     max_height = np.max(self.heights)
    #     if max_height >= 20 or self.L >= 20:
    #         return self.__repr__()
    #     elif max_height == 0:
    #         return str(np.zeros((1, self.L), dtype=int))
    #     else:
    #         matrix = np.zeros((max_height, self.L), dtype=int)
    #         for i in range(self.L):
    #             if self.heights[i] > 0:
    #                 matrix[-self.heights[i]:, i] = 1
    #         return f"The Oslo Model looks like:\n{str(matrix)}"
    #
    # def __repr__(self):
    #     return f"The z vector is {self.z}\nThe heights are {self.heights}"

    def is_system_stable(self) -> bool:
        """Check if the system is stable by an elementwise comparison of the z and z_threshold vectors.
        It is O(L) and may not always be the most efficient"""
        return np.all(self.z <= self.z_threshold)

    def reset(self) -> None:
        self.__init__(self.L)

    def run(self) -> None:
        """Run the model for 1 iteration by placing a grain on the leftmost site."""
        self._heights[0] += 1
        self.z[0] += 1

        # is_stable = self.is_system_stable()

        while not self.is_system_stable():
            """This loops brings the system to the next stable state. z and heights are hardcoded independently for
            performance reasons"""
            if self.z[0] > self.z_threshold[0]:
                self.z[0] -= 2
                self.z[1] += 1
                self._heights[0] -= 1
                self._heights[1] += 1
                self.z_threshold[0] = np.random.randint(1, 3)
            for i in range(1, self.L - 1):
                if self.z[i] > self.z_threshold[i]:
                    self.z[i] -= 2
                    self.z[i + 1] += 1
                    self.z[i - 1] += 1
                    self._heights[i] -= 1
                    self._heights[i + 1] += 1
                    self.z_threshold[i] = np.random.randint(1, 3)
            if self.z[-1] > self.z_threshold[-1]:
                self.z[-1] -= 1
                self.z[-2] += 1
                self._heights[-1] -= 1
                self.z_threshold[-1] = np.random.randint(1, 3)

        self.time += 1

    def unstable_indices(self):
        """Return the indices of the unstable sites"""
        return np.where(self.z > self.z_threshold)[0]

    def doctor(self):
        """Perform a few standard checks of common issues within the instance of OsloModel"""
        if not self.is_system_stable():
            warnings.warn("Current configuration is not stable")
        elif self.z != -np.diff(np.append(self._heights, 0)):
            warnings.warn("z and heights are not consistent. Check the run method")
        else:
            "No issues found"


if __name__ == "__main__":
    model = OsloModel(8)
