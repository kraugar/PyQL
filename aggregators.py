"""
Aggregators for PyQL
consider getting all of the classes here dynamically from query.py
    before buildng the lexer.
"""
import unittest


class Column(object):
    def __init__(self, **kwargs):
        self.data = kwargs.get('data', [])
        self.formatter = kwargs.get('format', str)

    def update(self, value, **kwargs):
        self.data.append(value)

    def value(self, **kwargs):
        return self.data


class Replace(Column):
    def __init__(self, **kwargs):
        self._value = None
        self.default = kwargs.get('default')
        self.formatter = kwargs.get('format', '%s')

    def update(self, value, **kwargs):
        if value is not None:
            self._value = value

    def value(self, **kwargs):
        return self._value


class Sum(Column):
    def __init__(self, **kwargs):
        self.data = kwargs.get('data', [])
        self.default = kwargs.get('default', 0)
        self.formatter = kwargs.get('format', '%d')
        self.N = kwargs.get('N')

    def update(self, value, **kwargs):
        if value is not None:
            self.data.append(value)

    def value(self, **kwargs):
        if not self.data:
            return self.default
        N = kwargs.get('N', self.N) or 0
        if len(self.data) < N:
            return None
        return sum(self.data[-N:])


class Average(Sum):
    def __init__(self, **kwargs):
        Sum.__init__(self, **kwargs)

    def value(self, **kwargs):
        if not self.data:
            return self.default
        N = self.N or 0
        len_data = len(self.data)
        if not len_data or len_data < N:
            return None
        ret = 1.*sum(self.data[-N:])/(N or len_data)
        return ret


AGGREGATOR_DICT = {'Sum': Sum, 'Average': Average,
                   'S': Sum, 'A': Average,
                   'Replace': Replace, 'Column': Column,
                   'R': Replace, 'C': Column}


class TestQuery(unittest.TestCase):

    def test_column(self):
        c = Column()
        c.update(3)
        c.update(4)
        self.assertEqual(c.value(), [3, 4])

    def test_replace(self):
        r = Replace()
        r.update(3)
        r.update(4)
        self.assertEqual(r.value(), 4)

    def test_sum(self):
        s = Sum()
        s.update(3)
        s.update(4)
        self.assertEqual(s.value(), 7)
        s = Sum(N=1)
        s.update(3)
        s.update(4)
        self.assertEqual(s.value(), 4)

    def test_average(self):
        a = Average()
        a.update(3)
        a.update(4)
        self.assertEqual(a.value(), 3.5)


if __name__ == '__main__':
    unittest.main()
