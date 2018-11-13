# Define tokens for the Pythonic Query Language.
# The PyQL translates a text query into python code.
# That code is executed for each row in the database.
# A PyQL query is exected in python:
#    any tokens not identified are passed right on through.
# Use regex rather than re to facilitate recurcive patterns.
#

from __future__ import print_function
import unittest

import sys
import ply.lex as lex
import regex as re
lex.re = re

tokens = [
    r'AT',               # @
    r'QUESTION_MARK',               # @
    r'COMMA',            # ,
    r'DOLLAR',           # $N indexes Nth column
    r'AS',               # Used to name a field or condition.
    r'CONJUNCTION',      # Delimits conditions
    r'PARAMETER',        # A database parameter.
    r'AGGREGATOR',       # An aggregator defined in PyQL.aggregators.
    r'PYTHON_FUNCTION',  # anything_else_that_looks_like_a_function(anything)
    r'COMPARATOR',       # A symbol like '<' (not an implicit group-by).
    r'DB_STRING',        # A database string (no need to quote).
    r'STRING',           # A string that was quoted.
    r'PYTHON',           # Everything else.
]

# want to access these from other modules
for t in tokens:
    exec("%s='%s'" % (t, t))


def t_STRING(t):
    r'(\'[^\']*\')|(\"[^\"]*\")'
    return t


def t_AT(t):
    r'[\s]*@[\s]*'
    return t


def t_QUESTION_MARK(t):
    r'\?'
    return t


def t_COMMA(t):
    r','
    return t


def t_DOLLAR(t):
    r'\$[0-9]+'
    return t


def t_AS(t):
    r'[\s]+as[\s]+'
    return t


# set by replacing this methods doc string.
def t_DB_STRING(t):
    r'$ replace this'
    return t


# set by replacing this methods doc string.
def t_PARAMETER(t):
    r'$ replace this'
    return t


# defined in PyQL.aggregators following the given pattern.
def t_AGGREGATOR(t):
    #            Aggregator          ( any_non_parens   anything_in_parans )     optional [explicit read key]
    r'(?P<AG1>\b[C|S|A|R|U])(?P<AG2>[\s]*\((?P<AG3>[^()]*)(?:(?&AG2)(?&AG3))*\))(?P<SB>\[(?:[^\[\]]++|(?&SB))*\])?'
    return t


def t_CONJUNCTION(t):
    r'[\s]+and[\s]+|[\s]+or[\s]+'
    return t


# so we can distinguish between groups and simple conditions
def t_COMPARATOR(t):
    r'!=|==|<=|>=|<|>|=|[\s]+is[\s]+not[\s]+|\bnot[\s]+in\b|[\s]+in[\s]+|[\s]+is[\s]+|\bnot[\s]+'
    return t


def t_PYTHON_FUNCTION(t):
    # see note for AGGREGATOR regex
    r'(?P<PF1>[_a-zA-Z]+[_a-zA-Z.0-9]*)*(?P<PF2>\((?P<PF3>[^()]*)(?:(?&PF2)(?&PF3))*\))'
    return t


def t_PYTHON(t):
    r'[^?]+?'
    return t


def t_error(t):
    raise Exception(r'Not too sure you got past "[^?]+?" with "%r"' % (t,))


def test(text):
    t_PARAMETER.__doc__ = r'team|hits|runs|errors|inning\ runs'
    t_DB_STRING.__doc__ = r'Cubs|Reds|Mets'
    lexer = lex.lex(debug=0)
    lexer.input(str(text))
    return [tok for tok in lexer]


class TestLexer(unittest.TestCase):

    def test_arguments(self):
        pyql = "hits?N=1,f=2"
        toks = test(pyql)
        self.assertEqual(toks[0].type, 'PARAMETER')
        self.assertEqual(toks[1].type, 'QUESTION_MARK')
        self.assertEqual(toks[5].type, 'COMMA')

    def test_aggregators(self):
        pyql = "A(runs),hits,S(errors) @ S(1@team)[team and qs[-1]>6] as 'Team' and hits<4"
        toks = test(pyql)
        #print("toks:", toks)
        self.assertEqual(toks[0].type, 'AGGREGATOR')
        self.assertEqual(toks[1].type, 'COMMA')
        self.assertEqual(toks[2].type, 'PARAMETER')
        self.assertEqual(toks[5].type, 'AT')
        self.assertEqual(toks[6].value[-3:], '>6]')


if __name__ == '__main__':
    unittest.main()
