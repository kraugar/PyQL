"""
the smallest unit is a term
between any two terms there can be:
  a COMMA - delimits fields and defines explicit query groups:
                                  a, b, c @ d>1, 2, 3
  a CONJUCTION - python in fields and acts as delimitor for conditions:
                                  a and b, c or d @ e and f=g
  a COMPARATOR - python in fields and distingishes a singleton from a group:
                                  a>b, c>d @ e and f=g
  another term - concatenate without query structural relevance
a field is a tuple of terms
a condition is a list of lists of tuples of terms,
and   query: fields[@conditions]? [\|query]* [\?fields]?
"""

from __future__ import print_function
import ply.yacc
import ply.lex
from lexer import tokens
import unittest


class Term(object):

    def __init__(self, value, as_term, flavor):
        self.value = value
        self.as_term = as_term or value
        self.flavor = flavor
        self.mc = None  # to be added in dt.

    def __repr__(self):
        if self.as_term:
            return '<T>value: `%s`, as: `%s`, flavor: `%s`</T>' % (
                self.value, self.as_term, self.flavor)


class Query(object):

    def __init__(self, fields, conditions=[]):
        self.arguments = []
        # strip white space in fields
        self.fields = []
        for field in fields:
            while not field[0].value.strip():
                field = field[1:]
            while not field[-1].value.strip():
                field = field[:-1]
            self.fields.append(field)
        self.conditions = conditions
        self.str_args = ','.join([''.join([tok.value for tok in argument])
                                             for argument in self.arguments])

    def cleave_args(self):
        # aggregator arguments are confused with comma delimited groups
        #   and I dont like the looks of S(points?N=2)
        cout = []
        args = ''
        for clist in self.conditions:
            ctmp = []
            for l, lst in enumerate(clist):
                if len(lst) == 1:
                    ctmp.append(lst)
                    continue
                if lst[0][0].flavor == 'PYTHON_FUNCTION':
                    egroups = []
                    fac = 1
                    for g, glist in enumerate(lst):
                        for t, term in enumerate(glist):
                            if term.flavor == 'PYTHON_FUNCTION':
                                if egroups:
                                    fac *= 2
                                    egroups.append(Term('+%d*'%(fac,), '', 'PYTHON'))
                                egroups.append(term)
                            else:
                                args += lst[g][0].value
                    ctmp.append([egroups])
                else:
                    ctmp.append([lst[0]])
                    args += lst[1][0].value
                for slist in clist[l+1:]:
                    args += ','.join([''.join([tok.value for tok in toks]) for toks in slist])
                break
            cout.append(ctmp[:])
        self.conditions = cout
        return args

    def __repr__(self):
        ret = "Fields:\n"
        for i, f in enumerate(self.fields):
            ret += ' f%d: %s\n' % (i+1, f)
        ret += "Conditions:\n"
        for i, c in enumerate(self.conditions):
            ret += ' c%d: %s len %d\n' % (i+1, c, len(c))
        if self.arguments:
            ret += "with str args: %s"%(self.str_args,)
        return ret


# As per Beazley: When parsing starts, try to make a "query" because it's
#      the name on the left-hand side of the first p_* function definition.
def p_query(p):
    """query : base_query
             | base_query QUESTION_MARK fields"""
    p[0] = p[1]
    if len(p) == 4:
        p[0].arguments = p[3]


def p_base_query(p):
    """base_query : fields
                  | fields AT conditions"""
    if len(p) == 2:
        p[0] = Query(fields=p[1])
    elif len(p) == 4:
        p[0] = Query(fields=p[1], conditions=p[3])


def p_fields(p):
    """fields : field
              | fields COMMA field"""
    if len(p) == 4:
        p[0] = p[1]+p[3:]
    else:
        p[0] = p[1:]


def p_field(p):
    """field : terms
             | field_conjunction_field
             | field_comparator_field"""
    p[0] = p[1]


def p_field_conjunction_field(p):
    """field_conjunction_field : field CONJUNCTION field"""
    p[0] = p[1] + (Term(p[2], None, 'CONJUNCTION'),) + p[3]


def p_field_comparator_field(p):
    """field_comparator_field : field COMPARATOR field"""
    p[0] = p[1] + (Term(p[2], None, 'COMPARATOR'),) + p[3]


def p_conditions(p):
    """conditions : condition
                  | conditions CONJUNCTION conditions"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1]+[[[(Term(p[2], None, 'CONJUNCTION'),)]]]+p[3]


def p_condition(p):
    """condition : terms
                 | comma_terms
                 | condition COMPARATOR condition"""
    if len(p) == 2:
        if type(p[1]) is tuple:
            # the extra list here is to normalize terms for conditions
            # group by have lists of len>1; singletone lists of len=1
            p[0] = [[p[1]]]
        else:
            p[0] = [p[1]]
    else:
        p[0] = p[1] + [[(Term(p[2], None, 'COMPARATOR'),)]] + p[3]


def p_python(p):
    """python : PYTHON
              | python PYTHON """
    p[0] = ''.join(p[1:])


def p_comma_terms(p):
    """comma_terms : terms COMMA terms
                   | comma_terms COMMA terms"""
    if type(p[1]) is list:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1], p[3]]


def p_terms(p):
    """terms : term"""
    p[0] = (p[1],)


def p_terms_term(p):
    """terms : terms term"""
    p[0] = p[1]+(p[2],)


def p_term_as(p):
    """term : term_as_string
            | term_as_python"""
    p[0] = p[1]


def p_term_as_python(p):
    """term_as_python : term AS python"""
    p[1].as_term = p[3]
    p[0] = p[1]


def p_term_as_string(p):
    """term_as_string : term AS STRING"""
    p[1].as_term = p[3][1:-1]
    p[0] = p[1]


# each flavor of term has a mapping for python metacode generation in dt.py
# Access database column by index (starting at $1).
def p_term_dollar(p):
    """term : DOLLAR"""
    p[0] = Term(p[1], None, 'DOLLAR')


# Find the database column by parameter name.
def p_term_parameter(p):
    """term : PARAMETER"""
    p[0] = Term(p[1], None, 'PARAMETER')


# Defined in aggregators.py. For example: Average, Sum, Unique.
def p_term_aggregtor(p):
    """term : AGGREGATOR"""
    p[0] = Term(p[1], None, 'AGGREGATOR')


# A single or double quoted string.
def p_term_string(p):
    """term : STRING"""
    p[0] = Term(p[1], None, 'STRING')


# Names that you do not want to bother quoting.
def p_term_db_string(p):
    """term : DB_STRING"""
    p[0] = Term(p[1], None, 'DB_STRING')


# A function or tuple or just (parens) not recognized above.
def p_term_python_function(p):
    """term : PYTHON_FUNCTION"""
    p[0] = Term(p[1], None, 'PYTHON_FUNCTION')


# Everything else.
def p_term_python(p):
    """term : python"""
    p[0] = Term(p[1], None, 'PYTHON')


def p_error(p):
    print("Syntax error in input: %s" % p)
    raise Exception("Syntax error in input: %s" % p)


def build_parser(debug=False):
    return ply.yacc.yacc(debug=debug)


def test(text):
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        filename="parselog.txt",
        filemode="w",
        format="%(filename)10s:%(lineno)4d:%(message)s")
    log = logging.getLogger()
    import lexer
    lexer.test(text)
    lexer.t_PARAMETER.__doc__ = r'team|hits|runs|errors|quarter\ scores|season'
    lexer.t_DB_STRING.__doc__ = r'Cubs|Reds|Mets'
    parser = build_parser(debug=True)
    qob = ply.yacc.parse(text, debug=log)
    return qob


class TestYaccer(unittest.TestCase):

    def test_arguments(self):
        pyql = "S(runs,N=1)@S(hits@team and season?N=2,format='%0.d')>10"
        qob = test(pyql)
        self.assertEqual(qob.fields[0][0].flavor, 'AGGREGATOR')

    def test_as(self):
        pyql = "S(runs,N=1) as 'Sruns'@S(hits@team and season?N=2,format='%0.d') as 'Shits'>10"
        qob = test(pyql)
        self.assertEqual(qob.fields[0][0].as_term, 'Sruns')
        self.assertEqual(qob.conditions[0][0][0][0].as_term, 'Shits')
        pyql = "(team.strip()) as T@1"
        qob = test(pyql)
        #print('qob',qob)
        self.assertEqual(qob.fields[0][0].as_term, 'T')


if __name__ == '__main__':
    unittest.main()
