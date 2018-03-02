from __future__ import print_function
import ply.lex as lex
import regex as re


# Define tokens for the Pythonic Query Language.
# The PyQL translates a text query into python code which executes the query.
# Any text query input which is not recognized as another token is tagged as python.

tokens = [
    "AT",               # @
    "COMMA",            # ,
    "AS",               # a keyword used to name a field or condition
    "CONJUNCTION",      # and, or
    "PARAMETER",        # a database parameter
    "AGGREGATOR",       # a database aggregator defined in PyQL.aggregators
    "PYTHON_FUNCTION",  # anything(anything)
    "COMPARATOR",       # a symbols that distinguishes a singleton query from a group-by
    "DB_STRING",        # a database string that you want to reference without quotes
    "STRING",           # a quoted string
    "PYTHON",           # everything else
]

def t_AT(t):
    r'@'
    return t

def t_COMMA(t):
    r','
    return t

def t_AS(t):
   r'[\s]+as[\s]+'
   return t

def t_STRING(t):
    r'(\'[^\']*\')|(\"[^\"]*\")'
    return t

# database strings will be entered here at run time by replacing this methods doc string.
def t_DB_STRING(t):
    r'$ replace this'
    return t

# database parameters will be entered here at run time by replacing this methods doc string.
def t_PARAMETER(t):
    r'$ replace this'
    return t

# aggregators defined in PyQL.aggregators will be entered here at run time.
def t_AGGREGATOR(t):
    r'(?P<AG1>\b[C|S|A|R])+(?P<AG2>\((?P<AG3>[^()]*+)(?:(?&AG2)(?&AG3))*\))'
    return t

def t_CONJUNCTION(t):
   r'[\s]+and[\s]+|[\s]+or[\s]+'
   return t

# so we can distinguish between groups and simple conditions
def t_COMPARATOR(t):
   r"!=|==|<=|>=|<|>|=|[\s]+is[\s]+not[\s]+|\bnot[\s]+in\b|[\s]+in[\s]+|[\s]+is[\s]+|\bnot[\s]+"
   return t

def t_PYTHON_FUNCTION(t):
    r'(?P<PF1>[_a-zA-Z]+[[_a-zA-Z.0-9]*)*(?P<PF2>\((?P<PF3>[^()]*+)(?:(?&PF2)(?&PF3))*\))'
    return t

def t_PYTHON(t):
    r".+?"
    return t

def t_error(t):
    raise TypeError("unknown text at %r" % (p.value,))

def test(text):
    t_PARAMETER.__doc__ = "team|hits|runs|errors"
    t_DB_STRING.__doc__ = "Cubs|Reds|Mets"
    lexer = lex.lex(debug=0)
    print("test:",text)
    lexer.input(str(text))
    while True:
        tok = lexer.token()
        if not tok: break
        print(tok.type,tok.value)

if __name__ == "__main__":
    test('A(runs,N=4,z=(1,2),format="%0.2f") as "Average Runs",(hits,errors)@team=Cubs and math.pow(hits,runs)<10')
