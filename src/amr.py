#!/usr/bin/env python2.7
#coding=utf-8
'''
Parser for Abstract Meaning Represention (AMR) annotations in Penman format.
A *parsing expression grammar* (PEG) for AMRs is specified in amr.peg
and the AST is built by the Parsimonious library (https://github.com/erikrose/parsimonious).
The resulting graph is represented with the AMR class.
When called directly, this script runs some cursory unit tests.

TODO: Include the smatch evaluation code 
(released at http://amr.isi.edu/evaluation.html under the MIT License).

@author: Nathan Schneider (nschneid@inf.ed.ac.uk)
@since: 2015-05-05
'''
from __future__ import print_function
import os, sys, re, fileinput, json
from pprint import pprint
from collections import defaultdict, namedtuple, Counter

from parsimonious.grammar import Grammar
from nltk.parse import DependencyGraph

def clean_grammar_file(s):
    return re.sub('\n[ \t]+', ' ', re.sub(r'#.*','',s.replace('\t',' ').replace('`','_backtick')))

with open('amr.peg') as inF:
    grammar = Grammar(clean_grammar_file(inF.read()))


class Var(object):
    def __init__(self, name):
        self._name = name
    def is_constant(self):
        return False
    def __repr__(self):
        return 'Var('+self._name+')'
    def __str__(self):
        return self._name
    def __eq__(self, that):
        return type(that)==type(self) and self._name==that._name
    def __hash__(self):
        return hash(repr(self))

class Concept(object):
    def __init__(self, name):
        self._name = name
    def is_constant(self):
        return False
    def __repr__(self):
        return 'Concept('+self._name+')'
    def __str__(self):
        return self._name
    def __eq__(self, that):
        return type(that)==type(self) and self._name==that._name
    def __hash__(self):
        return hash(repr(self))
        
class AMRConstant(object):
    def __init__(self, value):
        self._value = value
    def is_constant(self):
        return True
    def __repr__(self):
        return 'Const('+self._value+')'
    def __str__(self):
        return self._value
    def __eq__(self, that):
        return type(that)==type(self) and self._value==that._value
    def __hash__(self):
        return hash(repr(self))
        
class AMRString(AMRConstant):
    def __repr__(self):
        return '"'+self._value+'"'
    __str__ = __repr__

class AMRNumber(AMRConstant):
    def __repr__(self):
        return 'Num('+self._value+')'


class AMRError(Exception):
    pass

class AMRSyntaxError(Exception):
    pass

class AMR(DependencyGraph):
    '''
    An AMR annotation. Constructor parses the Penman notation. 
    Does not currently provide functionality for manipulating the AMR structure, 
    but subclassing from DependencyGraph does provide the contains_cycle() method.
    
    >>> s = """                                    \
    (b / business :polarity -                      \
       :ARG1-of (r / resemble-01                   \
                   :ARG2 (b2 / business            \
                             :mod (s / show-04)))) \
    """
    >>> a = AMR(s)
    >>> a
    (b / business :polarity -
        :ARG1-of (r / resemble-01
            :ARG2 (b2 / business
                :mod (s / show-04))))
    >>> a.reentrancies()
    Counter()
    >>> a.contains_cycle()
    False
    
    >>> a = AMR("(h / hug-01 :ARG0 (y / you) :ARG1 y :mode imperative)")
    >>> a
    (h / hug-01
        :ARG0 (y / you)
        :ARG1 y
        :mode imperative)
    >>> a.reentrancies()
    Counter({Var(y): 1})
    >>> a.contains_cycle()
    False
    
    >>> a = AMR('(h / hug-01 :ARG1 (p / person :ARG0-of h))')
    >>> a
    (h / hug-01
        :ARG1 (p / person
            :ARG0-of h))
    >>> a.reentrancies()
    Counter({Var(h): 1})
    >>> a.triples()     #doctest:+NORMALIZE_WHITESPACE
    [(Var(TOP), ':top', Var(h)), (Var(h), ':instance-of', Concept(hug-01)), 
     (Var(h), ':ARG1', Var(p)), (Var(p), ':instance-of', Concept(person)), 
     (Var(p), ':ARG0-of', Var(h))]
    >>> a.contains_cycle()
    [Var(p), Var(h)]
    
    >>> a = AMR('(h / hug-01 :ARG0 (y / you) :mode imperative \
    :ARG1 (p / person :ARG0-of (w / want-01 :ARG1 h)))')
    >>> # Hug someone who wants you to!
    >>> a.contains_cycle()
    [Var(w), Var(h), Var(p)]
    
    >>> a = AMR('(w / wizard    \
    :name (n / name :op1 "Albus" :op2 "Percival" :op3 "Wulfric" :op4 "Brian" :op5 "Dumbledore"))')
    >>> a
    (w / wizard
        :name (n / name
            :op1 "Albus"
            :op2 "Percival"
            :op3 "Wulfric"
            :op4 "Brian"
            :op5 "Dumbledore"))
    '''
    
    def __init__(self, anno):
        '''
        Given a Penman annotation string for a single rooted AMR, construct the data structure.
        Triples are stored internally in an order that preserves the layout of the 
        annotation (even though this doesn't matter for the "pure" graph).
        (Whitespace is normalized, however.)
        
        Will raise an AMRSyntaxError if notationally malformed, or an AMRError if 
        there is not a 1-to-1 mapping between (unique) variables and concepts.
        Will not check details such as the appropriateness of relation (role) names 
        or constants. Does not currently read or store metadata about the AMR.
        '''
        self._v2c = {}
        self._triples = []
        self._constants = set()
        
        self.nodes = defaultdict(lambda: {'address': None, 
                                          'type': None,
                                          'label': None,
                                          'head': None,
                                          'rel': None,  
                                          'deps': []})
        # Emulate the DependencyGraph (superclass) data structures somewhat.
        # There are some differences, e.g., in AMR it is possible for a node to have
        # multiple dependents with the same relation; so here, 'deps' is simply a list 
        # of dependents, not a mapping from relation types to dependents.
        
        TOP = Var('TOP')
        self.nodes[TOP]['address'] = TOP
        self.nodes[TOP]['type'] = 'TOP'
        if anno:
            self._anno = anno
            p = grammar.parse(anno)
            if p is None:
                raise AMRSyntaxError('Well-formedness error in annotation:\n'+anno.strip())
            self._analyze(p)
    
    def triples(self):  # overrides superclass implementation
        return self._triples
                
    def constants(self):
        return self._constants
        
    def concept(self, variable):
        return self._v2c[variable]
    
    def concepts(self):
        return self._v2c.items()
        
    def var2concept(self):
        return dict(self._v2c)
    
    def reentrancies(self):
        '''Counts the number of times each variable is mentioned in the annotation 
        beyond the one where it receives a concept. Non-reentrant variables are not 
        included in the output.'''
        c = defaultdict(int)
        for h, r, d in self.triples():
            if isinstance(d, Var):
                c[d] += 1
            elif isinstance(d, Concept):
                c[h] -= 1
        return Counter(c) + Counter()   # the addition removes non-positive entries
    
    #def __repr__(self):
    #    return 'AMR(v2c='+repr(self._v2c)+', triples='+repr(self._triples)+', constants='+repr(self._constants)+')'

    def __str__(self, compressed=False, indent=' '*4):
        '''Assumes triples are stored in a sensible order (reflecting how they are encountered in a valid AMR).'''
        s = ''
        stack = []
        instance_fulfilled = None
        for h, r, d in self.triples()+[(None,None,None)]:
            if r==':top':
                s += '(' + str(d)
                stack.append(d)
                instance_fulfilled = False
            elif r==':instance-of':
                s += ' / ' + str(d)
                instance_fulfilled  = True
            elif h==stack[-1] and r==':polarity':   # polarity gets to be on the same line as the concept
                s += ' ' + r + ' ' + str(d)
            else:
                while stack and h!=stack[-1]:
                    popped = stack.pop()
                    if instance_fulfilled is False:
                        # just a variable with no concept hanging off of it
                        # so we have an extra paren to get rid of
                        s = s[:-len(str(popped))-1] + str(popped)
                    else:
                        s += ')'
                    instance_fulfilled = None
                if d is not None:
                    s += '\n' + indent*len(stack) + r + ' (' + str(d)
                    stack.append(d)
                    instance_fulfilled = False
        return s

    __repr__ = __str__

    def _analyze(self, p):
        '''Analyze the AST produced by parsimonious.'''
        v2c = {}    # variable -> concept
        allvars = set() # all vars mentioned in the AMR
        elts = {}  # for interning variables, concepts, constants, etc.
        consts = set()  # all constants used in the AMR
    
        def intern_elt(x):
            return elts.setdefault(x, x)
    
        def walk(n):    # (v / concept...)
            triples = []
            deps = []
            v = None
            for ch in n.children:
                t = ch.expr_name
                if t=='VAR':
                    v = intern_elt(Var(ch.text))
                    allvars.add(v)
                elif t=='CONCEPT':
                    assert v is not None
                    if v in v2c:
                        raise AMRError('Variable has multiple concepts: '+str(v))
                    c = intern_elt(Concept(ch.text))
                    v2c[v] = c
                    self.add_node({'address': c, 'type': 'CONCEPT',
                                   'rel': ':instance-of', 'head': v, 'deps': []})
                    deps.append(c)
                    triples.append((v, ':instance-of', c))
                elif t=='' and ch.children:
                    for ch2 in ch.children:
                        _part, RELpart, _part, Ypart = ch2.children
                        rel = RELpart.text
                        assert rel is not None
                        assert len(Ypart.children)==1
                        q = Ypart.children[0]
                        tq = q.expr_name
                        n2 = None
                        triples2 = []
                        deps2 = []
                        if tq=='X':
                            n2, triples2, deps2 = walk(q)
                        elif tq=='NAMEDCONST':
                            n2 = intern_elt(AMRConstant(q.text))
                            consts.add(n2)
                        elif tq=='VAR':
                            n2 = intern_elt(Var(q.text))
                            allvars.add(n2)
                        elif tq=='STR':
                            n2 = intern_elt(AMRString(q.text[1:-1]))
                            consts.add(n2)
                        elif tq=='NUM':
                            n2 = intern_elt(AMRNumber(q.text))
                            consts.add(n2)
                        assert n2 is not None
                        self.add_node({'address': n2, 'type': tq,
                                       'rel': rel, 'head': v})
                        self.nodes[n2]['deps'].extend(deps2)
                        deps.append(n2)
                        triples.append((v, rel, n2))
                        triples.extend(triples2)
            return v, triples, deps
    
        assert p.expr_name=='ALL'
        
        n = None
        for ch in p.children:
            if ch.expr_name=='X':
                assert n is None    # only one top-level node per AMR
                n, triples, deps = walk(ch)
                self.add_node({'address': n, 'type': 'VAR', 
                               'rel': ':top', 'head': intern_elt(Var('TOP'))})
                self.nodes[n]['deps'].extend(deps)
                triples = [(intern_elt(Var('TOP')), ':top', n)] + triples
    
        if allvars - set(v2c.keys()):
            raise AMRError('Unbound variable(s): ' + ','.join(map(str,allvars - set(v2c.keys()))))
    
        # All is well, so store the resulting data
        self._v2c = v2c
        self._triples = triples
        self._constants = consts



good_tests = [
'''(h / hot)''',
'''(h / hot :mode expressive)''',
'''(h / hot :mode "expressive")''',
'''(h / hot :domain h)''',
'''  (  h  /  hot   :mode  expressive  )   ''',
'''  (  h  
/  
hot   
:mode  
expressive  
)   ''',
'''(h / hot
     :mode expressive
     :mod (e / emoticon
          :value ":)"))''',
'''(n / name :op1 "Washington")''',
'''(s / state :name (n / name :op1 "Washington"))''',
'''(s / state :name (n / name :op1 "Ohio"))
''',
'''(s / state :name (n / name :op1 "Washington") 
    )
''',
'''(s / state 
:name (n / name :op1 "Washington"))''',
'''(s / state :name (n / name :op1 "Washington") 
    :wiki "http://en.wikipedia.org/wiki/Washington_(state)")
''',
'''(f / film :name (n / name :op1 "Victor/Victoria") 
    :wiki "http://en.wikipedia.org/wiki/Victor/Victoria_(1995_film)")
''',
'''(g / go-01 :polarity -
      :ARG0 (b / boy))''',
'''(a / and
:op1 (l / love-01 :ARG0 (b / boy) :ARG1 (g / girl))
:op2 (l2 / love-01 :ARG0 g :ARG1 b)
)''',
'''(d / date-entity :month 2 :day 29 :year 2012 :time "16:30" :timezone "PST"
           :weekday (w / wednesday))''',
'''(a / and :op1 (d / day :quant 40) :op2 (n / night :quant 40))''',
'''(e / earthquake
           :location (c / country-region :wiki "Tōhoku_region"
                 :name (n / name :op1 "Tohoku"))
           :quant (s / seismic-quantity :quant 9.3)
           :time (d / date-entity :year 23 :era "Heisei"
                 :calendar (c2 / country :wiki "Japan"
                       :name (n2 / name :op1 "Japan"))))''',
''' (d / date-entity :polite +
           :time (a / amr-unknown))'''
]

sembad_tests = [    # not a syntax error, but malformed in terms of variables
'''(h / hot :mod (h / hot))''',
'''(h / hot :mod q)'''
]

bad_tests = [
'''h / hot :mode expressive''',
'''(hot :mode expressive)''',
'''(h/hot :mode expressive)''',
'''(h / hot :mode )''',
'''(h / hot :mode expressive''',
'''(h / hot :mode (expressive))''',
'''(h / hot :mode (e / ))''',
'''((h / hot :mode expressive)''',
'''(h / hot :mode expressive)

x''',
'''(s / state :name (n / name :op1 "  Washington  "))''',
'''(s / state :name (n / name :op1 "Washington") 

    )
''',
'''(s / state 

:name (n / name :op1 "Washington"))''',
'''(e / earthquake
           :location (c / country-region :wiki "Tōhoku_region"
                 :name (n / name :op1 "Tohoku"))
           :quant (s / seismic-quantity :quant 9.3.1)
           :time (d / date-entity :year 23 :era "Heisei"
                 :calendar (c2 / country :wiki "Japan"
                       :name (n2 / name :op1 "Japan"))))'''
]

def test():
    for good in good_tests:
        try:
            a = AMR(good)
        except AMRSyntaxError:
            print('Should be valid!')
            print(sembad)
        except AMRError:
            print('Should be valid!')
            print(sembad)

    for sembad in sembad_tests:
        try:
            a = AMR(sembad)
        except AMRSyntaxError:
            print('Parse should work!')
            print(sembad)
        except AMRError:
            pass    # should trigger exception
        else:
            print('Should be invalid!')
            print(sembad)

    for bad in bad_tests:
        try:
            a = AMR(bad)
        except AMRSyntaxError:
            pass
        else:
            print('Parse should fail!')
            print(bad)

if __name__=='__main__':
    test()
    import doctest
    doctest.testmod()
    