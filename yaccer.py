"""
the smallest unit is a term
between any two terms there can be:
  a comma - delimits fields and defines explicit query groups: a,b,c@d>1,2,3
  a CONJUCTION - delimits conditions and can also appear inside terms: a and b,c or d@e and f=g
  a COMPARATOR - distingishes a singleton from a group and can appear inside terms: a>b,c>d@e and f=g
  another term - concatenate without query structure relevance
a field is a list of terms
a condition can be a list of terms (for a singleton) or a list of list of terms (for an explicit group)
the query syntax is: fields@conditions
TODO:
parse guts of a function/aggregator
build mc
"""

from __future__ import print_function
import ply.yacc as yacc
import ply.lex as lex

from lexer import tokens

class Term(object):
    def __init__(self, term, as_term, flavor):
        self.term = term
        self.as_term = as_term
        self.flavor = flavor

    def add_as(self,as_term):
        self.as_term = as_term
        return self

    def __repr__(self):
        if self.as_term:
            return '<Term:%s AS %s (%s):mreT>'%(self.term, self.as_term,self.flavor)
        return '<Term:%s (%s):mreT>'%(self.term, self.flavor)

def ppretty(fs,cs):
    ret = "Fields\n"
    for f in fs:
        ret += 'f:%s\n'%(f,)
    ret += "\nConds\n"
    for c in cs:
        ret += 'c:%s\n'%(c,)
    return ret


# When parsing starts, try to make a "query" because it's
#      the name on left-hand side of the first p_* function definition.
def p_query(p):
    """query : fields AT conditions"""
    p[0] = ppretty(p[1], p[3])

def p_fields(p):
    """fields : field
       fields : fields COMMA field"""
    if len(p)==4:
        p[0] = p[1]+p[3:]
    else:
        p[0] = p[1:]

def p_field(p):
    """field : terms
       field : field CONJUNCTION field
       field : field COMPARATOR field"""
    if len(p) == 2:
        p[0]=p[1]
    else:
        p[0] = p[1] + [Term(p[2],None,'COMPARATOR')] + p[3]

def p_conditions(p):
    """conditions : condition
       conditions : conditions CONJUNCTION condition"""
    if len(p)==4:
        p[0] = p[1]+[p[2]]+p[3]
    else:
        p[0] = [p[1]]

def p_condition_singleton(p):
    """condition : terms COMPARATOR terms"""
    p[0] = [p[1]+[Term(p[2],None,'COMPARATOR')]+p[3]]

def p_condition_implicit_group(p):
    """condition : terms"""
    p[0] = [p[1]]

def p_cterms_or_terms(p):
    """cterms_or_terms : cterms
                       | terms"""
    p[0]=p[1]

def p_condition_explicit(p):
    """condition : cterms_or_terms COMPARATOR cterms_or_terms
       condition : cterms_or_terms COMPARATOR cterms_or_terms COMPARATOR cterms_or_terms"""
    if len(p) == 4:
        p[0] = [p[1],[Term(p[2],None,'COMPARATOR')],p[3]]
    elif len(p) == 6:
        p[0] = [p[1],[Term(p[2],None,'COMPARATOR')],p[3],[Term(p[4],None,'COMPARATOR')],p[5]]

def p_condition_implicit_explicit_group(p):
    """condition : cterms"""
    p[0] = [p[1]]

# no comparator
def p_terms(p):
    """terms : term"""
    p[0] = [p[1]]
def p_term_term(p):
    """terms : terms term"""
    p[0] = p[1]+p[2:]
def p_terms_comma_terms(p):
    """cterms : terms COMMA terms"""
    p[0] = [p[1],p[3]]
def p_cterms_comma_terms(p):
    """cterms : cterms COMMA terms"""
    p[0] = p[1]+[p[3]]

def p_term_as(p):
    """term : term AS STRING"""
    p[0] = p[1].add_as(p[3])

# no comma, no comparator
def p_term_parameter(p):
    """term : PARAMETER"""
    p[0] = Term(p[1],None,'PARAMETER')
def p_term_aggregtor(p):
    """term : AGGREGATOR"""
    p[0] = Term(p[1],None,'AGGREGATOR')
def p_term_string(p):
    """term : STRING"""
    p[0] = Term(p[1],None,'STRING')
def p_term_db_string(p):
    """term : DB_STRING"""
    p[0] = Term(p[1],None,'DB_STRING')
def p_term_python(p):
    """term : python"""
    p[0] = Term(p[1],None,'PYTHON')

def p_python_Python(p):
    """python : PYTHON"""
    p[0] = p[1]
def p_python_python_Python(p):
    """python : python PYTHON"""
    p[0] = p[1]+p[2]
def p_term_python_function(p):
    """term : PYTHON_FUNCTION"""
    p[0] = Term(p[1],None,'PYTHON_FUNCTION')

parser = yacc.yacc()

def test(text):
    import lexer
    lexer.test(text)
    lexer.t_PARAMETER.__doc__ = "team|hits|runs|errors|quarter\ scores"
    lexer.t_DB_STRING.__doc__ = "Cubs|Reds|Mets"
    lexer = lex.lex(module=lexer,debug=0)
    yacc.yacc()
    print(yacc.parse(text))


def test_suite():
    #test("runs+1+2+4/4@1")
    #test("runs>hits>errors/2.>runs/4,runs,hits,'errors'@runs")
    #test("10*S(runs)/2. as 'R'@A(runs,N=1,2,3)>3,4+runs and hits")
    #test("runs@hits>4")
    test("runs@10,9,8>hits>0,1,2,3 and 137>math.e")

if __name__ == '__main__':
    test_suite()
