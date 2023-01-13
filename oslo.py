import numpy as np
from tqdm import tqdm
import warnings


class OsloModel:
    """A class for the Oslo Model"""

    def __init__(self, L: int):
        self.L = L
        self._heights = np.zeros(L, dtype=int)
        self.__h = np.append(self._heights, 0)  # used for calculating z
        self.z_threshold = np.random.choice([1, 2], size=L, p=[0.5, 0.5])
        # This is a shortcut, because np.diff performs out[i] = a[i+1] - a[i]
        self.z = -np.diff(self.__h)
        self.time = 0

    @property
    def heights(self):
        return self._heights

    @heights.setter
    def heights(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("Heights must be a numpy array")
        if value.dtype != int:
            raise TypeError("Heights must be an array of integers")
        if value.ndim != 1:
            raise ValueError("Heights must be a 1D array")
        if value.shape[0] != self.L:
            raise ValueError("Heights must have the same length as L")
        self._heights = value
        self.__h = np.append(self._heights, 0)
        self.z = -np.diff(self.__h)

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
        return f"The z vector is {self.z} and the heights are {self._heights}"

    def is_system_stable(self) -> bool:
        """Check if the system is stable by an elementwise comparison of the z and z_threshold vectors.
        It is O(L) and may not always be the most efficient"""
        return np.all(self.z <= self.z_threshold)

    def reset(self) -> None:
        self.__init__(self.L)

    def relax(self, index):
        pass

    def run(self, n: int = 1) -> None:
        """Run the model for n iterations by placing a grain on the leftmost site. This assumes that the current
        configuration is stable."""
        for _ in tqdm(range(n), disable=(n < 100)):
            self._heights[0] += 1
            self.z[0] += 1

            is_stable = self.is_system_stable()

            while not self.is_system_stable():
                """This loops brings the system to the next stable state. z and heights are hardcoded independently for 
                performance reasons"""
                if self.z[0] > self.z_threshold[0]:
                    self.z[0] -= 2
                    self.z[1] += 1
                    self._heights[0] -= 1
                    self._heights[1] += 1
                    self.z_threshold[0] = np.random.choice([1, 2], p=[0.5, 0.5])
                for i in range(1, self.L - 1):
                    if self.z[i] > self.z_threshold[i]:
                        self.z[i] -= 2
                        self.z[i + 1] += 1
                        self.z[i - 1] += 1
                        self._heights[i] -= 1
                        self._heights[i + 1] += 1
                        self.z_threshold[i] = np.random.choice([1, 2], p=[0.5, 0.5])
                if self.z[-1] > self.z_threshold[-1]:
                    self.z[-1] -= 1
                    self.z[-2] += 1
                    self._heights[-1] -= 1
                    self.z_threshold[-1] = np.random.choice([1, 2], p=[0.5, 0.5])

        self.time += n

    def doctor(self):
        """Perform a few standard checks of common issues within the instance of OsloModel"""
        if not self.is_system_stable():
            warnings.warn("Current configuration is not stable")
        else:
            "No issues found"


if __name__ == "__main__":
    model = OsloModel(32)
    model.run(450)

