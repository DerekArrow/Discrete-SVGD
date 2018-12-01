import numpy as np;
from sortedcontainers import SortedSet as sset;
from factor import *

def readFileByTokens(path, specials=[]):
  """Helper function for parsing pyGM file formats"""
  import re
  spliton = '([\s'+''.join(specials)+'])'
  with open(path, 'r') as fp:
    for line in fp:
      #if line[-1]=='\n': line = line[:-1]
      tok = [t.strip() for t in re.split(spliton,line) if t and not t.isspace()]
      for t in tok: yield t

def readUai(filename):
  """Read in a collection (list) of factors specified in UAI (2006-?) format
  Example:
  >>> factor_list = readUai( 'path/filename.uai' )
  """
  dims = []           # store dimension (# of states) of the variables
  i = 0               # (local index over variables)
  cliques = []        # cliques (scopes) of the factors we read in
  factors = []        # the factors themselves

  gen = readFileByTokens(filename,'(),')   # get token generator for the UAI file
  type = next(gen)                   # read file type = Bayes,Markov,Sparse,etc
  nVar = int(next(gen))              # get the number of variables
  dims = [int(next(gen)) for i in range(nVar)] #   and their dimensions (states)
  nCliques = int(next(gen))          # get the number of cliques / factors
  cliques = [ None ] * nCliques
  for c in range(nCliques): 
    cSize = int(next(gen))           #   (size of clique)
    cliques[c] = [int(next(gen)) for i in range(cSize)]
  factors = [ None ] * nCliques 
  for c in range(nCliques):          # now read in the factor tables:
    tSize = int(next(gen))           #   (# of entries in table = # of states in scope)
    vs = VarSet(Var(v,dims[v]) for v in cliques[c])
    assert( tSize == vs.nrStates() )
    factorSize = tuple(dims[v] for v in cliques[c]) if len(cliques[c]) else (1,)
    tab = np.empty(tSize)
    for i in range(tSize): tab[i]=float(next(gen))
    tab = tab.reshape(factorSize)
    t2  = np.transpose(tab, tuple(np.argsort(cliques[c])))
    factors[c] = Factor(vs, np.array(t2,dtype=float,order=orderMethod))   # use 'orderMethod' from Factor class

  used = np.zeros((nVar,))
  for f in factors: used[f.v.labels] = 1
  for i in range(nVar):              # fill in singleton factors for any missing variables
    if dims[i] > 1 and not used[i]: factors.append(Factor([Var(i,dims[i])],1.0))

  return factors, nVar

