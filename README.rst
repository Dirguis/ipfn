ipfn
=======================

Iterative proportional fitting is an algorithm used is many different fields such as economics or social sciences, to alter results in such a way that aggregates along one or several dimensions match known marginals (or aggregates along these same dimensions).

For more information and examples, please visit:
- `wikipedia page on ipf <https://en.wikipedia.org/wiki/Iterative_proportional_fitting>`_
- `slides explaining the methodology and links to specific examples <http://www.demog.berkeley.edu/~eddieh/IPFDescription/AKDOLWDIPFTWOD.pdf>`_

----

The project is similar to the ipfp package available for R and tests have been run to ensure same results.

----

Example:
--------
Please, follow the example below to run the package. Several additional examples in addition to the one listed below, are listed in the ipfn.py script. This example is taken from `<http://www.demog.berkeley.edu/~eddieh/IPFDescription/AKDOLWDIPFTHREED.pdf>`_

First, let us define a matrix of N=3 dimensions, the matrix being of specific size 2*4*3 and populate that matrix with some values ::

    m = np.zeros((2,4,3))
    m[0,0,0] = 1
    m[0,0,1] = 2
    m[0,0,2] = 1
    m[0,1,0] = 3
    m[0,1,1] = 5
    m[0,1,2] = 5
    m[0,2,0] = 6
    m[0,2,1] = 2
    m[0,2,2] = 2
    m[0,3,0] = 1
    m[0,3,1] = 7
    m[0,3,2] = 2
    m[1,0,0] = 5
    m[1,0,1] = 4
    m[1,0,2] = 2
    m[1,1,0] = 5
    m[1,1,1] = 5
    m[1,1,2] = 5
    m[1,2,0] = 3
    m[1,2,1] = 8
    m[1,2,2] = 7
    m[1,3,0] = 2
    m[1,3,1] = 7
    m[1,3,2] = 6

Now, let us define some marginals. They all have to be less than N=3 dimensions and be consistent with the dimensions of contingency table m. For example, the marginal along the first dimension will be made of 2 elements. We want the sum of elements in m for dimensions 2 and 3 to equal the marginal::

    m[0,:,:].sum() == marginal[0]
    m[1,:,:].sum() == marginal[1]

The marginals are::

    xipp = np.array([52, 48])
    xpjp = np.array([20, 30, 35, 15])
    xppk = np.array([35, 40, 25])
    xijp = np.array([[9, 17, 19, 7], [11, 13, 16, 8]])
    xpjk = np.array([[7, 9, 4], [8, 12, 10], [15, 12, 8], [5, 7, 3]])

I used the letter p to denote the dimension(s) being summed over

Define the aggregates list and the corresponding list of dimension to indicate the algorithm which dimension(s) to sum over for each aggregate::

    aggregates = [xipp, xpjp, xppk, xijp, xpjk]
    dimensions = [[0], [1], [2], [0, 1], [1, 2]]

Finally, run the algorithm::

    for inc in range(10):
        m = ipfn().ipfn_np(m, aggregates, dimensions)
    print xijp[0,0]
    print m[0, 0, :].sum()
