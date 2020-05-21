from synthplayer.oscillators import Filter, Oscillator
from typing import Generator, List
from scipy.signal import butter, filtfilt


class LowPassFilter(Filter):
    """
    A low pass filter applied to an oscillator with the given cutoff
    """
    def __init__(self, source: Oscillator, cutoff: float, samplerate: float) -> None:
        assert isinstance(source, Oscillator)
        super().__init__([source])
        self.cutoff = cutoff
        self.samplerate

    def blocks(self) -> Generator[List[float], None, None]:
        source_blocks = self.sources[0].blocks()
        try:
            while True:
                block = next(source_blocks)
                b, a = butter(4, self.cutoff / (self.samplerate / 2.), 'low')
                yield filtfilt(b, a, block)
        except StopIteration:
            return
