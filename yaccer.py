# Define the grammar for the Pythonic Query Language.
# the smallest unit is a term
# between any two terms there can be:
#   a COMMA - delimits fields and defines explicit query groups:
#                                  a, b, c @ d>1, 2, 3
#   a CONJUCTION - python in fields and acts as delimitor for conditions:
#                                  a and b, c or d @ e and f=g
#   a COMPARATOR - python in fields and distingishes a singleton from a group:
#                                  a>b, c>d @ e and f=g
#   another term - concatenate without query structural relevance
# a field is a tuple of terms
# a condition is a list of lists of tuples of terms,
# a query is: fields@conditions?arguments


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
        self.mc = None  # the metacode: added in query.py

    def __repr__(self):
        if self.as_term:
            return '<T>value: `%s`, as: `%s`, flavor: `%s`</T>' % (
                self.value, self.as_term, self.flavor)


class Query(object):

    def __init__(self, fields, conditions=[], arguments=[]):
        self.arguments = arguments
        self.fields = []
        # strip white space in fields
        for field in fields:
            while not field[0].value.strip():
                field = field[1:]
            while not field[-1].value.strip():
                field = field[:-1]
            self.fields.append(field)
        self.conditions = conditions

    def __repr__(self):
        ret = "Fields:\n"
        for i, f in enumerate(self.fields):
            ret += ' f%d: %s\n' % (i+1, f)
        ret += "Conditions:\n"
        for i, c in enumerate(self.conditions):
            ret += ' c%d: %s len %d\n' % (i+1, c, len(c))
        if self.arguments:
            ret += "with arguments: %s" % (self.arguments,)
        return ret


# query internal to an Aggregator.
# no comma groups allowed here
#  which allows us to find arguments
# build parser with start=aggregator_query
def p_aggregator_query(p):
    """aggregator_query : fields
                  | fields AT singleton_conditions
                  | fields AT singleton_conditions COMMA fields"""
    if len(p) == 2:
        p[0] = Query(fields=p[1])
    elif len(p) == 4:
        p[0] = Query(fields=p[1], conditions=p[3])
    elif len(p) == 6:
        p[0] = Query(fields=p[1], conditions=p[3], arguments=p[5])


def p_singleton_conditions(p):
    """singleton_conditions : singleton_condition
                  | singleton_conditions CONJUNCTION singleton_conditions"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1]+[[[(Term(p[2], None, 'CONJUNCTION'),)]]]+p[3]


def p_singleton_condition(p):
    """singleton_condition : terms
                 | singleton_condition COMPARATOR singleton_condition"""
    if len(p) == 2:
        if type(p[1]) is tuple:
            # the extra list here is to normalize terms for conditions
            # group-bys have lists of len>1; singletons have lists of len=1
            p[0] = [[p[1]]]
        else:
            p[0] = [p[1]]
    else:
        p[0] = p[1] + [[(Term(p[2], None, 'COMPARATOR'),)]] + p[3]


# top-level query
# build parser with start=query
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
            # group-bys have lists of len>1; singletons have lists of len=1
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


# each flavor of term has a mapping for python metacode generation in query.py
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


def build_parser(debug=False, start='query'):
    return ply.yacc.yacc(debug=debug, start=start)


def test(text, in_aggregator=0):
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
    if in_aggregator:
        agg_parser = build_parser(debug=True, start='aggregator_query')
    else:
        parser = build_parser(debug=True, start='query')
    qob = ply.yacc.parse(text, debug=log)
    return qob


class TestYaccer(unittest.TestCase):

    def test_arguments(self):
        pyql = """S(runs,N=2) as SumRuns@S(hits@team and season,
                N=runs,format='%0.d')>10?output=scatter"""
        qob = test(pyql)
        self.assertEqual(qob.fields[0][0].flavor, 'AGGREGATOR')

    def test_in_aggregator(self):
        pyql = "points@team and 1,N=2,M=1"
        qob = test(pyql, in_aggregator=1)
        self.assertEqual(qob.arguments[0][0].value, 'N')
        pyql = "points@team and season=2010,N=2,M=1"
        qob = test(pyql, in_aggregator=1)
        self.assertEqual(qob.arguments[-1][0].value, 'M')

    def test_as(self):
        pyql = """S(runs,N=1) as 'Sruns'@S(hits@team and season,
                N=2,format='%0.d') as 'Shits'>10"""
        qob = test(pyql)
        self.assertEqual(qob.fields[0][0].as_term, 'Sruns')
        self.assertEqual(qob.conditions[0][0][0][0].as_term, 'Shits')
        pyql = "(team.strip()) as T@1"
        qob = test(pyql)
        self.assertEqual(qob.fields[0][0].as_term, 'T')


if __name__ == '__main__':
    # qob=test('runs@runs=3,output=scatter if S(1)>2 else table')
    # qob=test('runs@team and runs=3,output=scatter,N=int(math.pow(hits,0.5))')
    # qob=test("points@team and season=2010,N=2,M=1",in_aggregator=1)
    # print('qob.fields',qob)
    unittest.main()
