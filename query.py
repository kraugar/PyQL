from __future__ import print_function
import pandas as pd
import ply
import lexer
import yaccer
import aggregators
import itertools
import regex
from collections import OrderedDict
# import modules for use inside of the query loop
import math
import re
import random
import unittest
import string

class CacheDict(OrderedDict):
    """Used inside of dt query loop to hold a dictionary of
       column instance keyed by query group."""
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
            if len(cond_terms) == len(cond_terms[0]) == 1 and cond_terms[0][0].flavor == lexer.CONJUNCTION:
                igroup = False
            # parenthesis used merely to isolate does not make an implicit group
            elif len(cond_terms) == len(cond_terms[0]) == 1 and cond_terms[0][0].flavor == lexer.PYTHON_FUNCTION:
                func, args = cond_terms[0][0].value[:-1].split('(', 1)
                # a terminal comma would make it a tuple, not an implicit group
                if not func and args.strip()[-1] != ",":
                    pyql = "%s" % (args,)
                    qob = self.parser.parse(pyql, self.lexer)
                    if len(qob.fields)==1 and [t for t in qob.fields[0] if t.flavor == lexer.COMPARATOR]:
                        igroup = False
            # a comparator indicates a singleton condition, not an implicit group
            elif [c for c in cond_terms if c[0].flavor == lexer.COMPARATOR]:
                igroup = False
            if igroup:
                mc_cond += ('True',)
                mc_group = ''.join([''.join([tok.as_term for tok in term]) for term in cond_terms]).strip()
                mc_val = ''.join([''.join([tok.mc for tok in term]) for term in cond_terms]).strip()
                mc_key += (r'r"""%s = %%s"""%%(%s,)'%(mc_group,mc_val),)
            else:
                mc_cond += (''.join([''.join([tok.mc for tok in term]) for term in cond_terms]), )
                mc_key += (r'r"""%s"""'%(''.join([''.join([tok.as_term for tok in term]) for term in cond_terms])),)
        # sub queries often do not have conditons: set defaults
        mc_cond = mc_cond or ('1',)
        mc_key = mc_key or ('1',)
        self.mc = (r'''(%s,) if %s else None'''%('+'.join(mc_key).strip(),' '.join(mc_cond).strip()))
        self.cmc = compile(self.mc, 'metacode', 'eval')

class PYQL(object):

    def __init__(self, df, **kwargs):
        self.bar_pat = re.compile(r'''((?:[^\|"']|"[^"]*"|'[^']*')+)\|(.*)''')
        self.non_index_num_pat = regex.compile("(?<!\[)\d+(?!\])")
        self.df = df
        self.kwargs = kwargs
        self.build_ply()
        self.context = None  # where we are in the top level parse: field | condition

    def build_ply(self):
        params = [p.strip() for p in self.df.columns.values.tolist()]
        #print("building ply with params:",params)
        # sort by length to match longest first
        params.sort(key=lambda x:-len(x))
        # negative look ahead to not match part of a longer string
        #    or start of a function
        re_params = '|'.join(map(lambda x: x + r'(?!(?:[a-zA-Z_]|[\s]*\())', params))
        re_params = re_params.replace(r' ', r'\ ')
        lexer.t_PARAMETER.__doc__ = re_params
        # db_strings are strings we don't want to bother quoting
        if self.kwargs.get(r'db_strings'):
            db_strings = r'|'.join(self.kwargs[r'db_strings'])
            db_strings = db_strings.replace(r' ', r'\ ')
            lexer.t_DB_STRING.__doc__ = db_strings
        self.lexer = ply.lex.lex(module=lexer)
        self.parser = yaccer.build_parser(debug=False)

    def add_metacode_to_term(self, term):
        """Add term.mc.
           Expand iteratively for aggregators and python functions.
           Aggregators build up self.cache."""
        context = self.context # fields and conditions are handled differently
        if term.flavor == 'DOLLAR':
            term.mc = 'self.df.iloc[_i][%d]' % (int(term.value[1:]) - 1)
        elif term.flavor == 'PARAMETER':
            col = self.df.columns.get_loc(term.value)
            term.mc = 'self.df.iloc[_i][%d]' % col
        elif term.flavor in ['PYTHON','COMPARATOR'] and term.value == '=':
            term.mc = '=='
        elif term.flavor in ['STRING', 'PYTHON']:
            term.mc = term.value
        elif term.flavor == 'DB_STRING':
            term.mc = r'r"""%s"""' % (term.value,)
        elif term.flavor in ['PYTHON_FUNCTION']:
            func, args = term.value[:-1].split('(',1)
            args = args.strip()
            add_comma_back = ''
            qob_fields = []
            if len(args):
                if args[-1] == ",":
                    add_comma_back = ","
                    args = args.strip()[:-1]
                pyql = "%s@1"%(args,)
                qob = self.parser.parse(pyql, self.lexer)
                [self.expand_field(field) for field in qob.fields]
                for field in qob.fields:
                    if not self.is_pure_aggregator(field):
                        term.not_pure_aggregator = 1
                qob_fields = qob.fields
            term.mc = "%s(%s%s)"%(func,','.join([''.join([t.mc for t in f]) for f in qob_fields]),add_comma_back)

        elif term.flavor in ['AGGREGATOR']:
            value =  term.value.strip()
            if value[-1] == ']':
                value, square_bracket, foo = regex.split('(?P<sb>\[(?:[^\[\]]++|(?&sb))*\])',value)
                value = value.strip()
            else:
                square_bracket = False
            agg,pyql = value[:-1].split('(',1)
            term.aggregator = agg.strip() # to init the parent field
            qob = self.parser.parse(pyql, self.lexer)
            cargs = qob.str_args
            if not cargs: # not  `?args` format; perhaps ,args
                cargs = qob.cleave_args()
                conditions = self.init_conditions(qob.conditions)
            [self.expand_field(field) for field in qob.fields[:1]]
            if len(qob.fields)>1:
                nvps = []
                for field in qob.fields[1:]:
                    nvp = ''
                    for tok in field:
                        nvp += tok.value
                    nvps.append(nvp)
                cargs = ",".join(nvps)
            field_mcs = [''.join([t.mc for t in f]) for f in qob.fields[:1]]
            if context == 'condition':
                # the _key of a condition cache is given by that aggregator's own conditions
                conditions = self.init_conditions(qob.conditions)
                mc_key = ' '.join([c.mc for c in conditions])
            else:
                # the _key of a field cache is accessed by `_key`
                #    which is built up in mc from condtions
                mc_key = "_key"
            setdefault_args = ','.join(['%s'%(mc_key,)]+field_mcs[1:]) # used to update
            if not square_bracket:
                # read and write at the same key
                get_args = mc_key
            else:
                sb_pyql = '1@%s'%(square_bracket.strip()[1:-1] ,) # parse like a condition
                sb_qob = self.parser.parse(sb_pyql, self.lexer)
                conditions = self.init_conditions(sb_qob.conditions)
                get_args = ' '.join([c.mc for c in conditions])
            update_value = field_mcs[0]
            term.cache_offset = len(self.cache)
            term.mc = "1*self.cache[%s].get(%s).value()"%(
                            term.cache_offset,
                            get_args)
            self.cache.append(CacheDict(aggregators.AGGREGATOR_DICT[term.aggregator]))
            self.cache[-1].mcup = "self.cache[%s].setdefault(%s,%s).update(%s)"%(
                            term.cache_offset,
                            setdefault_args,cargs,
                            update_value)
            self.cache[-1].cmcup = compile(self.cache[-1].mcup,'metacode','exec')
        else:
            term.mc = term.value

    def expand_field(self, field):
        # add metacode to each term and append self.cache as needed
        [self.add_metacode_to_term(term) for term in field]

    def is_pure_aggregator(self,field):
        #print('pure?',field)
        for term in field:
            if hasattr(term,'not_pure_aggregator'):
                return 0
            if term.flavor in ['PYTHON'] and regex.search(self.non_index_num_pat,term.value):
                #  1*S(points) gives a running average
                return 0
            if term.flavor not in ['PYTHON_FUNCTION', 'AGGREGATOR','PYTHON','STRING']:
                return  0
        return 1

    def init_fields(self,fields):
        fout = []
        for field in fields:
            [self.add_metacode_to_term(term) for term in field]
            mc = ''.join([term.mc for term in field]).strip()
            name = ''.join([term.as_term for term in field]).strip()
            qfield = CacheDict(aggregators.Column,name=name)
            qfield.pure_aggregators = self.is_pure_aggregator(field)
            qfield.mc = mc
            qfield.cmc = compile(qfield.mc,'metacode',"eval") # add error handling
            fout.append(qfield)
        return fout

    def init_conditions(self,conditions):
        cout = [] #
        # first attach meta code and init cache for all terms
        for condition in conditions:
            for groups in condition:
                [[self.add_metacode_to_term(term) for term in group] for group in groups]
            cout.append(list(itertools.product(*condition)))
        # return a list of ConditionLists.
        # The metacode for each ConditionList is generated here on init
        return [ConditionList(cond,self.parser,self.lexer) for cond in itertools.product(*cout)]

    def reduce(self,res):
        for k,key in enumerate(res.keys()):
            if not k:
                df = res[key]
            else:
                df = df.append(res[key],ignore_index=True)
        return df

    def query(self,pyql,**kwargs):
        """ handle subsequent queries"""
        mo = re.search(self.bar_pat,pyql)
        if mo:   pyql,subsequent = mo.group(1),mo.group(2)
        else:    subsequent = None
        res = self.single_query(pyql,**kwargs)
        if subsequent is None:
            return res
        rres = self.reduce(res)
        if not subsequent.strip():
            return rres
        add_pyql(rres)
        return rres.pyql(subsequent,**kwargs)

    def single_query(self, pyql, **kwargs):
        """pass in a text query of form: `fields @ conditions and return a dataframe
             with a datatable of fields for each conditional group"""
        debug = kwargs.get('debug',0)
        verbose = kwargs.get('verbose',0)
        self.cache = []
        self.qob = self.parser.parse(pyql, self.lexer)
        self.context = 'field'
        fields = self.init_fields(self.qob.fields)
        n_field_cache = len(self.cache)
        self.context = 'condition'
        conditions = self.init_conditions(self.qob.conditions)
        # for each row in the database
        mc = '' #the query looop'
        mc += "for _i in range(%d):\n"%(self.df.shape[0],)
        mc += "\t_keys = []\n" # collect the condition keys
        # for each explicit, that is comma-delimited, conditional group
        for cond in conditions:
            mc += "\ttry:\n\t\t_keys.append(%s)\n"%cond.mc
            if debug:
                mc += "\texcept Exception as ex:\n\t\tprint('error in query loop condition: %s'%ex)\n"
            else:
                mc += "\texcept: pass\n"
        if verbose:
            mc += "\tprint('_keys:',_keys)\n"
        mc += "\t_keys = [' '.join(_k) for _k in _keys if _k is not None]\n"
        if n_field_cache:
            mc += "\t#upate field cache\n"
            mc += "\tfor _key in _keys:\n"
            for fc in self.cache[:n_field_cache]:
                mc += "\t\ttry: %s\n"%fc.mcup
                if debug:
                    mc += "\t\texcept Exception as ex: print('error in field cache update: %s'%ex)\n"
                else:
                    mc += "\t\texcept: pass\n"
        mc += "\t#update results\n"
        mc += "\tfor _key in _keys:\n"
        for f,field in enumerate(fields):
            if field.pure_aggregators: # get the value of these after the loop
                mc += "\t\tfields[%d].setdefault(_key)\n"%(f,)
            else:
                mc += "\t\ttry:\n"
                mc += "\t\t\t_val = %s\n"%(field.mc,)
                if debug:
                    mc += "\t\texcept Exception as ex:\n"
                    mc += "\t\t\tprint('error in field value: %s'%ex)\n"
                else:
                    mc += "\t\texcept:\n"
                mc += "\t\t\t_val = None\n"
                mc += "\t\tfields[%d].setdefault(_key).update(_val)\n"%(f,)
        mc += "\t #update any cache conditions at their own keys\n"
        for cc in self.cache[n_field_cache:]:
            mc += "\ttry:\n\t\t%s\n"%cc.mcup
            if debug:
                mc += "\texcept Exception as ex:\n\t\tprint('error in update condition cache:%s'%ex)\n"
            else:
                mc += "\texcept: pass\n"
        if verbose:
            for i,line in enumerate(mc.split('\n')):
                print("%s: %s"%(i+1,line))
        cmc = compile(mc,'metacode','exec')
        exec(cmc)
        # build a dictionary of data frame indexed by _key
        result = {}
        # if all fields are aggregators, we need to init the dataframe with index=[0]
        index = [0]
        # _key is common to text query, metacode and realcode.
        for _key in fields[0].keys():
            data = OrderedDict()
            for field in fields:
                name = field.name
                if field.pure_aggregators:
                    #print('post query loop evalig for pure_agg',field.mc)
                    try: val = [eval(field.cmc)]
                    except: val = [None]
                else:
                    index = None
                    val = field[_key].value()
                data[name] = val
            #result[_key] = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
            #result[_key] = pd.DataFrame(data,index=index)
            result[_key] = pd.DataFrame.from_dict(data,orient='index').transpose()

        if verbose:
            print("cache:\n\c",'\n\n'.join(["%s"%(c,) for c in self.cache]))

        return result

def add_pyql(df, **kwargs):
    df.pyql = PYQL(df, **kwargs).query

def sample_df():
    d = {'date':[20171019,20171021,20171024,20171026,20171028],
         'team':['Bulls','Bulls','Bulls','Bulls','Bobs'],
         'opponent':['Raptors','Spurs','Cavaliers','Hawks','Thunder'],
         'points':[100,77,112,91,69],
         'opponent points':[79,101,93,94,101],
         'quarter scores':[[23, 14, 27, 36],[21, 17, 17, 22],[38, 30, 24, 20],[18, 19, 28, 26],[23, 8, 22, 16]]
         }
    df =  pd.DataFrame(d)
    return df

def test(pyql):
    df = sample_df()
    add_pyql(df, db_strings=['Bulls', 'Bears', 'Bobs'])
    return df.pyql(pyql)
    #df.pyql('date,"DA-"+team@team=Bulls and points-opponent points>10,12,14 and quarter scores[0]<25,50')
    # any depth of Python is easy
    #df.pyql('date, team, math.pow(points,max(min(quarter scores[0],2),1.)) @ team==Bulls')
    #print(df.pyql("S(points),points@1"))
    #sys.exit(0)
    #print(df.pyql("_key,(S(points) as SP,A(math.pow(sum(quarter scores[:2]),2))) as WTQ@A(points@team)>40",debug=1))
    res = df.pyql("points,(points),(S(points)*S(points)) as SP2,S(2*points)@1",verbose=0)['1 = 1']
    print('r:',res)
    #sys.exit(0)

    print( df.pyql("points@S(points@points>100 and 'Fred')[points>100 and 'Fred']>20",debug=1,verbose=1))
    print(df.pyql("points@S(points@2>1 and S(23>2@S(2@S(1)>2))>200 and team,N=2,X='bob')>1"))
    df.pyql("points@S(points@team and points>10,N=2)[team and points>10]>110")
    df.pyql("date,points,S(points*(S(points)>10)),1*S(points)/A(max(quarter scores[-2:])) @(S(points@team)>100),(S(points?N=2)>200),'bob'")
    res = df.pyql("points@S(points@(team[0]='B'),(team='Bulls'),(team[-1]='s'),N=2,f=4)>0")
    res = df.pyql("points@A(S(S(1@1,N=2)=2,N=2,F='U'),F=67)",debug=0)
    res = df.pyql("A(points),S(points*(S(1,N=1)>0),N=1)@team",debug=0,verbose=0)
    #res = df.pyql("points as 'Pts',team as 'T'@(points<100 and S(1)>1) as 'O100'",debug=1,verbose=1)
    print('res:',res)
    # any nasty nesting is easy
    res= df.pyql("team ,   1 * S ( points ), A(  min ( points , S( max   ( quarter scores ),N=2))?bob='knob') @ team + 'foo' and S(points@team)>1")
    print(res)


class TestQuery(unittest.TestCase):

    def test_simple(self):
        pyql = "points,team@1"
        res = test(pyql)
        self.assertEqual(res['1 = 1'].iloc[0][0], 100)
        self.assertEqual(res['1 = 1'].iloc[-1][-1], 'Bobs')

    def test_aggregator(self):
        pyql = "A(points),S(points,N=2)@team"
        res = test(pyql)
        self.assertEqual(res['team = Bulls'].iloc[0][0], 95)
        self.assertEqual(res['team = Bulls'].iloc[0][1], 203)
        pyql = "(1*A(points),1*S(points,N=2)),points@1"
        res = test(pyql)
        self.assertEqual(res['1 = 1'].iloc[1][0], (88.5, 177) )
        self.assertEqual(res['1 = 1'].iloc[1][1], 77)

    def test_subsequent(self):
        pyql = "A(points),S(points,N=2),R(_key)@team|"
        res = test(pyql)
        self.assertEqual(res.iloc[0][0], 69)
        pyql = "R(date),A(points),S(points,N=2),R(_key)@team|"
        res = test(pyql)
        self.assertEqual(res.iloc[0][1], 69)
        pyql = " R ( date ) as D , A (points) as AP,S(points,N=2) as SP,(R(_key).split('=')[-1].strip()) as T@team|(AP+SP) as C1@T=Bulls |"
        res = test(pyql)
        self.assertEqual(res.iloc[0][0], 298)


if __name__ == '__main__':
    #pyql = "R(_key).split(' ')[-1]@team|"
    #res = test(pyql)
    #print(res)
    unittest.main()
