"""
Aggregators for PyQL
probably want to get all of the classes here dynamically from dt.py:
"""



class Column(object):
    def __init__(self,**kwargs):
        self.data = kwargs.get('data',[])
        self.formatter = kwargs.get('format',str)

    def update(self,value,**kwargs):
        self.data.append(value)

    def value(self,**kwargs):
        return self.data

    def update_and_return(self,value,**kwargs):
        context = kwargs.get('context','field')
        if context == 'condition':
            ret_value = self.value(**kwargs)
            print('cond',ret_value)
            self.update(value,**kwargs)
        else:
            self.update(value,**kwargs)
            ret_value = self.value(**kwargs)
        return ret_value

class Replace(Column):
    def __init__(self,**kwargs):
        self._value = None
        self.default = kwargs.get('default')
        self.formatter = kwargs.get('format','%s')

    def update(self,value,**kwargs):
        if value is not None:
            self._value = value

    def value(self,**kwargs):
        return self._value

class Sum(Column):
    def __init__(self,**kwargs):
        self.data = kwargs.get('data',[])
        self.default = kwargs.get('default')
        self.formatter = kwargs.get('format','%d')
        self.N = kwargs.get('N')

    def update(self,value,**kwargs):
        if value is not None:
            self.data.append(value)

    def value(self,**kwargs):
        if not self.data:
            return self.default
        N = kwargs.get('N',self.N) or 0
        if len(self.data) < N:
            return None
        return sum(self.data[-N:])

class Average(Sum):
    def __init__(self,**kwargs):
        Sum.__init__(self,**kwargs)

    def value(self,**kwargs):
        if not self.data:
            return self.default
        N = self.N or 0
        len_data = len(self.data)
        if not len_data or len_data < N:
            return None
        ret = 1.*sum(self.data[-N:])/(N or len_data)
        return ret

AGGREGATOR_DICT = {'S':Sum,'A':Average,'R':Replace,'C':Column}


def test_sum():
    s = Sum(N=3)
    s.update(3)
    s.update(4)
    print("ur",s.update_and_return(7,context='condition'))
    print('sum',s.value(N=3))

if __name__ == '__main__':
    #test_sum()
    test_column_dict()
