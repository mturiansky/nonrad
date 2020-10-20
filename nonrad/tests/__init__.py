# pylint: disable=all

from pathlib import Path

TEST_FILES = Path(__file__).absolute().parent / '..' / '..' / 'test_files'


class FakeAx:
    def plot(*args, **kwargs):
        pass

    def scatter(*args, **kwargs):
        pass

    def set_title(*args, **kwargs):
        pass

    def axhline(*args, **kwargs):
        pass

    def set_xlim(*args, **kwargs):
        pass

    def set_ylim(*args, **kwargs):
        pass

    def set_xlabel(*args, **kwargs):
        pass

    def set_ylabel(*args, **kwargs):
        pass

    def set_yscale(*args, **kwargs):
        pass

    def fill_between(*args, **kwargs):
        pass


class FakeFig:
    def subplots(self, x, y, **kwargs):
        return [FakeAx() for _ in range(y)]
