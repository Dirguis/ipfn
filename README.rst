ipfn
=======================

Iterative proportional fitting is an algorithm used is many different fields such as economics or social sciences, to alter results in such a way that aggregates along one or several dimensions match known marginals (or aggregates along these same dimensions).

For more information and examples, please visit:
- `wikipedia page on ipf <https://en.wikipedia.org/wiki/Iterative_proportional_fitting>`_
- `slides explaining the methodology and links to specific examples <http://www.demog.berkeley.edu/~eddieh/IPFDescription/AKDOLWDIPFTWOD.pdf>`_

----

The project is similar to the ipfp package available for R and tests have been run to ensure same results.

----

Instruction to run the package:

Please, follow the example below to run the package. Several additional examples in addition to the one listed below, are listed in the ipfn.py script.

First, let us define a matrix of N=3 dimensions, the matrix being of specific size 2*4*3
    m = np.zeros((2,4,3))

First, define the aggregates or marginals. They should be at most of dimensions N-1. Then, list
