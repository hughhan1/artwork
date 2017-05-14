import math

class Interval():
    def __init__(self, low, high):
        self.low = low
        self.high = high
    
    def __init__(self, intervalstring):
        values = intervalstring.split('-')
        self.low = values[0]
        self.high = values[1]

    def contains(value):
        return self.low <= value and value < self.high

    @staticmethod
    def range(value):
        """
        Returns the interval range associated with this value, rounded to the
        nearest 10.
        """
        return int(math.floor(value / 10.0)) * 10, int(math.ceil(value / 10.0)) * 10
    
    @staticmethod
    def range_str(value):
        """
        Returns the interval range associated with this value, rounded to the
        nearest 10.
        """
        low, high = Interval.range(value)
        return str(low) + '-' + str(high)
    
    @staticmethod
    def time_period(value):
        return 'Modern' if value < 1946 else 'Contemporary'
