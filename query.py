from __future__ import print_function
import pandas as pd
import ply
import lexer
import yaccer
import aggregators
import itertools
import regex
import collections
import unittest
#  modules for use inside of the query loop
import math
import re
import random
import string


class CacheDict(collections.OrderedDict):
    """Used inside of dt query loop to hold a dictionary of
       aggregator instance keyed by query group."""
    def __init__(self, aggregator, name=''):
        super(CacheDict, self).__init__()
        self.name = name
        self.aggregator = aggregator

    def setdefault(self, key, **kwargs):
        if key is None:
            return
        ret = self.get(key)
        if ret is None:
            ret = self[key] = self.aggregator(**kwargs)
        return ret


class ConditionList(list):
    """ a list of conditional terms.
        builds metacode      """
    def __init__(self, t, parser, lexer):
        super(ConditionList, self).__init__(list(t or []))
        self.parser = parser
        self.lexer = lexer
        self.add_metacode()

    def add_metacode(self):
        mc_cond = ()  # the python
        mc_key = ()   # the `as` terms
        for cond_terms in self:
            # Explicit groups are handled in init_conditions.
            #  Here we need to find implicit groups
            igroup = True
            #  a conjunction is not an impicit groups
            if (len(cond_terms) == len(cond_terms[0]) == 1 and
                    cond_terms[0][0].flavor == lexer.CONJUNCTION):
                igroup = False
            # parentheses used merely to isolate
            #  does not make an implicit group
            elif (len(cond_terms) == len(cond_terms[0]) == 1
                  and cond_terms[0][0].flavor == lexer.PYTHON_FUNCTION):
                func, args = cond_terms[0][0].value[:-1].split('(', 1)
                # a terminal comma makes a tuple, not an implicit group
                if not func and args.strip()[-1] != ",":
                    pyql = "%s" % (args,)
                    qob = self.parser.parse(pyql, self.lexer)
                    if (len(qob.fields) == 1 and
                        [t for t in qob.fields[0]
                         if t.flavor == lexer.COMPARATOR]):
                        igroup = False
            # a comparator indicates a singleton condition,
            #  not an implicit group
            elif [c for c in cond_terms if c[0].flavor == lexer.COMPARATOR]:
                igroup = False
            if igroup:
                mc_cond += ('True',)
                mc_group = ''.join([''.join([tok.as_term for tok in term])
                                    for term in cond_terms]).strip()
                mc_val = ''.join([''.join([tok.mc for tok in term])
                                  for term in cond_terms]).strip()
                mc_key += (r'r"""%s = %%s"""%%(%s,)' % (mc_group, mc_val),)
            else:
                mc_cond += (''.join([''.join([tok.mc for tok in term])
                                     for term in cond_terms]), )
                mc_key += (r'r"""%s"""' % (
                    ''.join([''.join([tok.as_term for tok in term])
                             for term in cond_terms])),)
        # sub queries often do not have conditons: set defaults
        mc_cond = mc_cond or ('1',)
        mc_key = mc_key or ('1',)
        self.mc = (r'''(%s,) if %s else None''' % (
                   '+'.join(mc_key).strip(),
                   ' '.join(mc_cond).strip()))
        self.cmc = compile(self.mc, 'metacode', 'eval')


class PYQL(object):

    def __init__(self, df, **kwargs):
        self.bar_pat = re.compile(r'''((?:[^\|"']|"[^"]*"|'[^']*')+)\|(.*)''')
        self.non_index_num_pat = regex.compile(r'(?<!\[)\d+(?!\])')
        self.df = df
        self.kwargs = kwargs
        self.build_ply()
        # uninitiated Sums are 0 (Averages throw an error)
        self.default_sum = aggregators.Sum()
        # where we are in the top level parse: field | condition
        self.context = None

    def build_ply(self):
        params = [p.strip() for p in self.df.columns.values.tolist()]
        # sort by length to match longest first
        params.sort(key=lambda x: -len(x))
        # negative look ahead to not match part of a longer string
        #    or start of a function
        re_params = '|'.join(
            map(lambda x: x + r'(?!(?:[a-zA-Z_]|[\s]*\())', params))
        re_params = re_params.replace(r' ', r'\ ')
        lexer.t_PARAMETER.__doc__ = re_params
        # db_strings are strings we don't want to bother quoting
        if self.kwargs.get(r'db_strings'):
            db_strings = self.kwargs['db_strings']
            db_strings.sort(key=lambda x: -len(x))
            re_db_strings = r'|'.join(db_strings)
            re_db_strings = re_db_strings.replace(r' ', r'\ ')
            lexer.t_DB_STRING.__doc__ = re_db_strings
        self.lexer = ply.lex.lex(module=lexer)
        self.parser = yaccer.build_parser(debug=0, start='query')
        self.aggregator_parser = yaccer.build_parser(
            debug=0, start='aggregator_query')

    def add_metacode_to_term(self, term):
        """Add term.mc.
           Expand iteratively for aggregators and python functions.
           Aggregators build up self.cache."""
        context = self.context  # fields and conditions are handled differently
        if term.flavor == 'DOLLAR':
            # get column by offset
            term.mc = 'self.df.iloc[_i][%d]' % (int(term.value[1:]) - 1)
        elif term.flavor == 'PARAMETER':
            # get column by name
            col = self.df.columns.get_loc(term.value)
            term.mc = 'self.df.iloc[_i][%d]' % col
        elif term.flavor in ['PYTHON', 'COMPARATOR'] and term.value == '=':
            term.mc = '=='
        elif term.flavor in ['STRING', 'PYTHON']:
            term.mc = term.value
        elif term.flavor == 'DB_STRING':
            term.mc = r'r"""%s"""' % (term.value,)
        elif term.flavor in ['PYTHON_FUNCTION']:
            func, args = term.value[:-1].split('(', 1)
            args = args.strip()
            add_comma_back = ''
            qob_fields = []
            if len(args):
                if args[-1] == ",":
                    add_comma_back = ","
                    args = args.strip()[:-1]
                pyql = "%s@1" % (args,)
                qob = self.parser.parse(pyql, self.lexer)
                [self.expand_field(field) for field in qob.fields]
                for field in qob.fields:
                    if not self.is_pure_aggregator(field):
                        term.not_pure_aggregator = 1
                qob_fields = qob.fields
            term.mc = "%s(%s%s)" % (
                func,
                ','.join([''.join([t.mc for t in f]) for f in qob_fields]),
                add_comma_back)
        elif term.flavor in ['AGGREGATOR']:
            value = term.value.strip()
            if value[-1] == ']':
                value, square_bracket = regex.split(
                    r'(?P<sb>\[(?:[^\[\]]++|(?&sb))*\])', value)[:2]
                value = value.strip()
            else:
                square_bracket = False
            agg, pyql = value[:-1].split('(', 1)
            term.aggregator = agg.strip()
            if term.aggregator in ['S', 'Sum']:
                # uninitiated Sums have default.value() = 0
                cache_default = 'self.default_sum'
            else:
                # uninited averages (etc) fail
                cache_default = None
            qob = self.aggregator_parser.parse(pyql, self.lexer)
            conditions = qob.conditions
            if len(qob.fields) > 1:
                qob.fields, qob.arguments = qob.fields[:1], qob.fields[1:]
            [self.expand_field(field) for field in qob.fields]
            if qob.arguments:
                #  allows a db param `N` and a summative arg `N`: S(N,N=hits)
                [self.expand_field(field, preserve_first=2)
                 for field in qob.arguments]
                cargs = ','.join([''.join([t.mc or t.value for t in f])
                                  for f in qob.arguments])
            else:
                cargs = ''
            field_mcs = [''.join([t.mc for t in f]) for f in qob.fields[:1]]
            if context == 'condition':
                # the _key of a condition cache is given by that
                #   aggregator's own conditions
                conditions = self.init_conditions(conditions)
                mc_key = ' '.join([c.mc for c in conditions])
            else:
                # the _key of a field cache is accessed by `_key`
                #    which is built up in mc from condtions
                mc_key = "_key"
            setdefault_args = ','.join(['%s' % (mc_key, )] + field_mcs[1:])
            if not square_bracket:
                # read and write at the same key
                get_args = mc_key
            else:
                sb_pyql = '1@%s' % (square_bracket.strip()[1:-1], )
                sb_qob = self.parser.parse(sb_pyql, self.lexer)
                conditions = self.init_conditions(sb_qob.conditions)
                get_args = ' '.join([c.mc for c in conditions])
            update_value = field_mcs[0]
            term.cache_offset = len(self.cache)
            term.mc = "1*self.cache[%s].get(%s,%s).value()" % (
                            term.cache_offset,
                            get_args, cache_default)
            self.cache.append(CacheDict(
                          aggregators.AGGREGATOR_DICT[term.aggregator]))
            self.cache[-1].mcup = "self.cache[%s].setdefault(%s,%s)" % (
                               term.cache_offset, setdefault_args, cargs)
            self.cache[-1].mcup += ".update(%s)" % (update_value,)
        else:
            term.mc = term.value

    def expand_field(self, field, preserve_first=0):
        # add metacode to each term and append self.cache as needed
        [self.add_metacode_to_term(term) for term in field[preserve_first:]]

    def is_pure_aggregator(self, field):
        for term in field:
            if hasattr(term, 'not_pure_aggregator'):
                return 0
            if term.flavor in ['PYTHON'] and regex.search(
                    self.non_index_num_pat, term.value):
                #  1*S(points) gives a running average
                return 0
            if term.flavor not in ['PYTHON_FUNCTION',
                                   'AGGREGATOR', 'PYTHON', 'STRING']:
                return 0
        return 1

    def init_fields(self, fields):
        fout = []
        for field in fields:
            self.expand_field(field)
            mc = ''.join([term.mc for term in field]).strip()
            name = ''.join([term.as_term for term in field]).strip()
            qfield = CacheDict(aggregators.Column, name=name)
            qfield.pure_aggregators = self.is_pure_aggregator(field)
            qfield.mc = mc
            qfield.cmc = compile(qfield.mc, 'metacode', "eval")
            fout.append(qfield)
        return fout

    def init_conditions(self, conditions):
        cout = []
        # first attach meta code and init cache for all terms
        for condition in conditions:
            for groups in condition:
                [[self.add_metacode_to_term(term) for term in group]
                 for group in groups]
            cout.append(list(itertools.product(*condition)))
        # return a list of ConditionLists.
        # The metacode for each ConditionList is generated on init
        return [ConditionList(cond, self.parser, self.lexer)
                for cond in itertools.product(*cout)]

    def reduce(self, res):
        for k, key in enumerate(res.keys()):
            if not k:
                df = res[key]
            else:
                df = df.append(res[key], ignore_index=True)
        return df

    def query(self, pyql, **kwargs):
        """ handle subsequent queries"""
        mo = re.search(self.bar_pat, pyql)
        if mo:
            pyql, subsequent = mo.group(1), mo.group(2)
        else:
            subsequent = None
        res = self.single_query(pyql, **kwargs)
        if subsequent is None:
            return res
        rres = self.reduce(res)
        if not subsequent.strip():
            return rres
        add_pyql(rres)
        return rres.pyql(subsequent, **kwargs)

    def single_query(self, pyql, **kwargs):
        """pass in a text query of form: `fields @ conditions
             and return an OrderedDict
             with keys given by conditional groups
             and dataframe of fields for values"""
        debug = kwargs.get('debug', 0)
        verbose = kwargs.get('verbose', 0)
        self.cache = []
        self.qob = self.parser.parse(pyql, self.lexer)
        self.context = 'field'
        fields = self.init_fields(self.qob.fields)
        n_field_cache = len(self.cache)
        self.context = 'condition'
        conditions = self.init_conditions(self.qob.conditions)
        mc = ''  # the query looop'
        mc += "for _i in range(%d):\n" % (self.df.shape[0],)
        mc += "\t_keys = []\n"  # collect the condition keys
        #  for each explicit, (ie comma-delimited), conditional group
        for cond in conditions:
            mc += "\ttry:\n\t\t_keys.append(%s)\n" % cond.mc
            if debug:
                mc += "\texcept Exception as ex:\n"
                mc += "\t\tprint('error in query loop condition: %s'%ex)\n"
            else:
                mc += "\texcept: pass\n"
        if verbose:
            mc += "\tprint('_keys:',_keys)\n"
        mc += "\t_keys = [' '.join(_k) for _k in _keys if _k is not None]\n"
        if n_field_cache:
            mc += "\t#  upate field cache\n"
            mc += "\tfor _key in _keys:\n"
            for fc in self.cache[:n_field_cache]:
                mc += "\t\ttry: %s\n" % fc.mcup
                if debug:
                    mc += "\t\texcept Exception as ex:\n"
                    mc += "\t\t\tprint('error in field cache update: %s'%ex)\n"
                else:
                    mc += "\t\texcept: pass\n"
        mc += "\t  #update results\n"
        mc += "\tfor _key in _keys:\n"
        for f, field in enumerate(fields):
            if field.pure_aggregators:
                #  no reference to the value of this field in query
                mc += "\t\tfields[%d].setdefault(_key)\n" % (f,)
            else:
                mc += "\t\ttry:\n"
                mc += "\t\t\t_val = %s\n" % (field.mc,)
                if debug:
                    mc += "\t\texcept Exception as ex:\n"
                    mc += "\t\t\tprint('error in field value: %s'%ex)\n"
                else:
                    mc += "\t\texcept:\n"
                mc += "\t\t\t_val = None\n"
                mc += "\t\tfields[%d].setdefault(_key).update(_val)\n" % (f,)
        mc += "\t#  update any cache conditions at their own keys\n"
        for cc in self.cache[n_field_cache:]:
            mc += "\ttry:\n\t\t%s\n" % cc.mcup
            if debug:
                mc += "\texcept Exception as ex:\n"
                mc += "\t\tprint('error in update condition cache:%s'%ex)\n"
            else:
                mc += "\texcept: pass\n"
        if verbose:
            for i, line in enumerate(mc.split('\n')):
                print("%s: %s" % (i+1, line))
        cmc = compile(mc, 'metacode', 'exec')
        exec(cmc)
        # build a dictionary of data frames indexed by _key
        result = collections.OrderedDict()
        # if all fields are aggregators, init the dataframe with index=[0]
        index = [0]
        # _key is common to text query, metacode and realcode.
        for _key in fields[0].keys():
            data = collections.OrderedDict()
            for field in fields:
                name = field.name
                if field.pure_aggregators:
                    try:
                        val = [eval(field.cmc)]
                    except Exception:
                        val = [None]
                else:
                    index = None
                    val = field[_key].value()
                data[name] = val
            result[_key] = pd.DataFrame.from_dict(data,
                                                  orient='index').transpose()

        if verbose:
            print("cache:\n\n", '\n\n'.join(["%s" % (c,) for c in self.cache]))
        return result


def add_pyql(df, **kwargs):
    df.pyql = PYQL(df, **kwargs).query


def sample_df():
    d = {'date': [20171019, 20171021, 20171024, 20171026, 20171028],
         'team': ['Bulls', 'Bulls', 'Bulls', 'Bulls', 'Bobs'],
         'opponent': ['Raptors', 'Spurs', 'Cavaliers', 'Hawks', 'Thunder'],
         'points': [100, 77, 112, 91, 69],
         'opponent points': [79, 101, 93, 94, 101],
         'quarter scores': [[23, 14, 27, 36],
                            [21, 17, 17, 22],
                            [38, 30, 24, 20],
                            [18, 19, 28, 26],
                            [23, 8, 22, 16]]
         }
    df = pd.DataFrame(d)
    return df


def nba_test(pyql):
    df = sample_df()
    add_pyql(df, db_strings=['Bulls', 'Bears', 'Hawks'])
    return df.pyql(pyql, verbose=1, debug=1)


class TestQuery(unittest.TestCase):

    def test_simple(self):
        pyql = "points,team@1"
        res = nba_test(pyql)
        self.assertEqual(res['1 = 1'].iloc[0][0], 100)
        self.assertEqual(res['1 = 1'].iloc[-1][-1], 'Bobs')

    def test_aggregator(self):
        pyql = "A(points),S(points,N=2)@team"
        res = nba_test(pyql)
        self.assertEqual(res['team = Bulls'].iloc[0][0], 95)
        self.assertEqual(res['team = Bulls'].iloc[0][1], 203)
        pyql = "points@Sum(1,N=(points=100) + 2)=3"
        res = nba_test(pyql)
        self.assertEqual(res['Sum(1,N=(points=100) + 2)=3'].iloc[0][0], 91)
        pyql = "(1*A(points),1*S(points,N=2)),points@1"
        res = nba_test(pyql)
        self.assertEqual(res['1 = 1'].iloc[1][0], (88.5, 177))
        self.assertEqual(res['1 = 1'].iloc[1][1], 77)
        pyql = "(1*A(points),1*S(points,foo='bar',N=2*(points=100))),points@1"
        res = nba_test(pyql)
        self.assertEqual(res['1 = 1'].iloc[1][0], (88.5, 177))
        self.assertEqual(res['1 = 1'].iloc[1][1], 77)

    def test_subsequent(self):
        pyql = "A(points),S(points,N=2),R(_key)@team|"
        res = nba_test(pyql)
        self.assertEqual(res.iloc[0][0], 95)
        pyql = "R(date),A(points),S(points,N=2),R(_key)@team|"
        res = nba_test(pyql)
        self.assertEqual(res.iloc[0][2], 203)
        pyql = """R ( date ) as D , A (points) as AP,S(points,N=2) as SP,
            (R(_key).split('=')[-1].strip()) as T
             @team|(AP+SP) as C1@T[-3:-1]='ll'|"""
        res = nba_test(pyql)
        self.assertEqual(res.iloc[0][0], 298)

    def test_nested(self):
        pyql = "A(points),S(points,N=2),R(_key)@team and S(1@S(1@team))<1|"
        res = nba_test(pyql)
        print('r', res)
        self.assertEqual(res.iloc[1][0], 69)


if __name__ == '__main__':
    # pyql = "points@Sum(1,N=(points=100) + 2)=3"
    # res = nba_test(pyql)
    # print('r:',res)
    unittest.main()
