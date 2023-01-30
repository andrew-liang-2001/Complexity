import numpy as np
from numba.experimental import jitclass  # https://numba.readthedocs.io/en/stable/user/jitclass.html
from numba import int32, int64

L = np.array([4, 8, 16, 32, 64, 128, 256])  # define L in the namespace


@jitclass(spec=[
    ('L', int32),
    ('_heights', int32[:]),
    ('time', int64),  # time may be quite big so int64 is a safe choice
    ("z_threshold", int32[:]),
    ("z", int32[:]),
])
class OsloModel:
    """A class for the Oslo Model"""
    def __init__(self, L: int):
        """Note that under the numba implementation, creation of multiple models is slow because of the need for
        random arrays to be looped over."""
        self.L = L
        self._heights = np.zeros(L, dtype=np.int32)
        temp = np.zeros(L, dtype=np.int32)
        for i in range(L):
            temp[i] = np.random.randint(1, 3)
        self.z_threshold = temp
        temp2 = np.append(self._heights, np.array(0, dtype=np.int32))
        self.z = temp2[:-1] - temp2[1:]
        self.time = 0

    @property
    def heights(self):
        return self._heights

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

    def is_system_stable(self) -> bool:
        """Check if the system is stable by an elementwise comparison of the z and z_threshold vectors.
        It is O(L) and may not always be the most efficient"""
        return np.all(self.z <= self.z_threshold)

    def reset(self) -> None:
        self.__init__(self.L)

    def run(self) -> int:
        """Run the model for a single iteration by placing a grain on the leftmost site. Return the avalanche size."""
        self._heights[0] += 1
        self.z[0] += 1
        s = 0

        while not self.is_system_stable():
            """This loops brings the system to the next stable state. z and heights are hardcoded independently for
            performance reasons"""
            if self.z[0] > self.z_threshold[0]:
                self.z[0] -= 2
                self.z[1] += 1
                self._heights[0] -= 1
                self._heights[1] += 1
                self.z_threshold[0] = np.random.randint(1, 3)
                s += 1
            for i in range(1, self.L - 1):
                if self.z[i] > self.z_threshold[i]:
                    self.z[i] -= 2
                    self.z[i + 1] += 1
                    self.z[i - 1] += 1
                    self._heights[i] -= 1
                    self._heights[i + 1] += 1
                    self.z_threshold[i] = np.random.randint(1, 3)
                    s += 1
            if self.z[-1] > self.z_threshold[-1]:
                self.z[-1] -= 1
                self.z[-2] += 1
                self._heights[-1] -= 1
                self.z_threshold[-1] = np.random.randint(1, 3)
                s += 1

        self.time += 1
        return s

    def unstable_indices(self):
        """Return the indices of the unstable sites"""
        return np.where(self.z > self.z_threshold)[0]

    # def doctor(self):
    #     """Perform a few standard checks of common issues within the instance of OsloModel"""
    #     temp2 = np.append(self._heights, np.array(0, dtype=np.int32))
    #     if not self.is_system_stable():
    #         print("Current configuration is not stable")
    #     elif not np.all(self.z, -np.diff(temp2)):
    #         print("z and heights are not consistent. Check the run method")
    #     else:
    #         "No issues found"


if __name__ == "__main__":
    model = OsloModel(16)
