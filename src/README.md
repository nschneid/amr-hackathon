amr.py provides an API for reading AMRs, i.e., semantic annotations for sentences 
in the [Abstract Meaning Representation](http://amr.isi.edu/) that have been 
specified using the PENMAN notation. The `AMR` class serves as a data structure 
in which variables, concepts, constants, and other elements can be accessed. 
Functionality such as pretty-printing and cycle-checking is also provided:

    >>> a = AMR('(h / hug-01 :ARG1 (p / person :ARG0-of h))')
    
    >>> a
    (h / hug-01
        :ARG1 (p / person
            :ARG0-of h))
    
    >>> a.reentrancies()
    Counter({Var(h): 1})
    
    >>> a.triples()
    [(Var(TOP), ':top', Var(h)), (Var(h), ':instance-of', Concept(hug-01)),
     (Var(h), ':ARG1', Var(p)), (Var(p), ':instance-of', Concept(person)),
     (Var(p), ':ARG0-of', Var(h))]
    
    >>> a.contains_cycle()
    [Var(h), Var(p)]

If present, ISI-style alignments to words in the sentence are accessible as well. 

    >>> a = AMR("(h / hug-01~e.2 :polarity~e.1 -~e.1 :ARG0 (y / you~e.3) :ARG1 y \
                 :mode~e.0 imperative~e.5 :result (s / silly-01~e.4 :ARG1 y))", \
                "Do n't hug yourself silly !".split())
    >>> a
    (h / hug-01~e.2[hug] :polarity~e.1[n't] -~e.1[n't]
        :ARG0 (y / you~e.3[yourself])
        :ARG1 y
        :mode~e.0[Do] imperative~e.5[!]
        :result (s / silly-01~e.4[silly]
            :ARG1 y))

(See the doctests in amr.py for further examples.)

This code relies on the [parsimonious](https://github.com/erikrose/parsimonious) 
library for parsing the AMR annotations. The grammar is specified in amr.peg. 

Contributors:

- Nathan Schneider (@nschneid)
- Daniel Hershcovich (@danielhers)

Bug spotters:

- Jonathan May (@jonmay)
- Marco Damonte (@mdtux89)
