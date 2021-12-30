ipfn
=======================

Iterative proportional fitting is an algorithm used is many different fields such as economics or social sciences, to alter results in such a way that aggregates along one or several dimensions match known marginals (or aggregates along these same dimensions).

The algorithm exists in 2 versions:

*   numpy version, which the fastest by far
*   pandas version, which is much slower but easier to use than the numpy version


The algorithm recognizes the input variable type and and uses the appropriate version to solve the problem. To install the package:

*   pip install ipfn
*   pip install git+http://github.com/dirguis/ipfn@master

For more information and examples, please visit:

*   `wikipedia page on ipf <https://en.wikipedia.org/wiki/Iterative_proportional_fitting>`_
*   `slides explaining the methodology and links to specific examples <http://www.demog.berkeley.edu/~eddieh/IPFDescription/AKDOLWDIPFTWOD.pdf>`_
*   https://github.com/Dirguis/ipfn

----

If you want to test the package, clone the repo and from the main folder, run:

*   py.test --verbose --color=yes tests/tests.py

----

The project is similar to the ipfp package available for R and tests have been run to ensure same results.

----

Input Variables:
  * original: numpy darray matrix or dataframe to perform the ipfn on.
  * aggregates: list of numpy array or darray or pandas dataframe/series. The aggregates are the same as the marginals.
They are the target values that we want along one or several axis when aggregating along one or several axes.
  * dimensions: list of lists with integers if working with numpy objects, or column names if working with pandas objects.
Preserved dimensions along which we sum to get the corresponding aggregates.
  * convergence_rate: if there are many aggregates/marginal, it could be useful to loosen the convergence criterion.
  * max_iteration: Integer. Maximum number of iterations allowed.
  * verbose: integer 0, 1 or 2. Each case number includes the outputs of the previous case numbers.

    * 0: Updated matrix returned.

    * 1: Flag with the output status (0 for failure and 1 for success).

    * 2: dataframe with iteration numbers and convergence rate information at all steps.

  * rate_tolerance: float value. If above 0.0, like 0.001, the algorithm will stop once the difference between the conv_rate variable of 2 consecutive iterations is below that specified value.


Wikipedia example with Numpy:
-----------------------------
To illustrate Iterative Proportional Fitting, Wikipedia uses an example `here <https://en.wikipedia.org/wiki/Iterative_proportional_fitting#Example>`_

Below is that example solved with IPFN::

    import numpy as np
    from ipfn import ipfn

    m = [[40, 30, 20, 10], [35, 50, 100, 75], [30, 80, 70, 120], [20, 30, 40, 50]]
    m = np.array(m)
    xip = np.array([150, 300, 400, 150])
    xpj = np.array([200, 300, 400, 100])

    aggregates = [xip, xpj]
    dimensions = [[0], [1]]

    IPF = ipfn.ipfn(m, aggregates, dimensions, convergence_rate=1e-6)
    m = IPF.iteration()
    print(m)


Example with the numpy version of the algorithm:
------------------------------------------------
Please, follow the example below to run the package. Several additional examples in addition to the one listed below, are listed in the ipfn.py script. This example is taken from `<http://www.demog.berkeley.edu/~eddieh/IPFDescription/AKDOLWDIPFTHREED.pdf>`_

First, let us define a matrix of N=3 dimensions, the matrix being of specific size 2*4*3 and populate that matrix with some values ::

    from ipfn import ipfn
    import numpy as np
    import pandas as pd

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

Now, let us define some marginals::

  xipp = np.array([52, 48])
  xpjp = np.array([20, 30, 35, 15])
  xppk = np.array([35, 40, 25])
  xijp = np.array([[9, 17, 19, 7], [11, 13, 16, 8]])
  xpjk = np.array([[7, 9, 4], [8, 12, 10], [15, 12, 8], [5, 7, 3]])

I used the letter p to denote the dimension(s) being summed over

For this specific example, they all have to be less than N=3 dimensions and be consistent with the dimensions of contingency table m. For example, the marginal along the first dimension will be made of 2 elements. We want the sum of elements in m for dimensions 2 and 3 to equal the marginal::

    m[0,:,:].sum() == xipp[0]
    m[1,:,:].sum() == xipp[1]

Define the aggregates list and the corresponding list of dimension to indicate the algorithm which dimension(s) to sum over for each aggregate::

    aggregates = [xipp, xpjp, xppk, xijp, xpjk]
    dimensions = [[0], [1], [2], [0, 1], [1, 2]]

Finally, run the algorithm::

    IPF = ipfn.ipfn(m, aggregates, dimensions)
    m = IPF.iteration()
    print(xijp[0,0])
    print(m[0, 0, :].sum())


Example with the pandas version of the algorithm:
-------------------------------------------------
In the same fashion, we can run a similar example, but using a dataframe::

    from ipfn import ipfn
    import numpy as np
    import pandas as pd

    m      = np.array([1., 2., 1., 3., 5., 5., 6., 2., 2., 1., 7., 2.,
                   5., 4., 2., 5., 5., 5., 3., 8., 7., 2., 7., 6.], )
    dma_l  = [501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501,
              502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502]
    size_l = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
              1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]

    age_l  = ['20-25','30-35','40-45',
              '20-25','30-35','40-45',
              '20-25','30-35','40-45',
              '20-25','30-35','40-45',
              '20-25','30-35','40-45',
              '20-25','30-35','40-45',
              '20-25','30-35','40-45',
              '20-25','30-35','40-45']

    df = pd.DataFrame()
    df['dma'] = dma_l
    df['size'] = size_l
    df['age'] = age_l
    df['total'] = m

    xipp = df.groupby('dma')['total'].sum()
    xpjp = df.groupby('size')['total'].sum()
    xppk = df.groupby('age')['total'].sum()
    xijp = df.groupby(['dma', 'size'])['total'].sum()
    xpjk = df.groupby(['size', 'age'])['total'].sum()
    # xppk = df.groupby('age')['total'].sum()

    xipp.loc[501] = 52
    xipp.loc[502] = 48

    xpjp.loc[1] = 20
    xpjp.loc[2] = 30
    xpjp.loc[3] = 35
    xpjp.loc[4] = 15

    xppk.loc['20-25'] = 35
    xppk.loc['30-35'] = 40
    xppk.loc['40-45'] = 25

    xijp.loc[501] = [9, 17, 19, 7]
    xijp.loc[502] = [11, 13, 16, 8]

    xpjk.loc[1] = [7, 9, 4]
    xpjk.loc[2] = [8, 12, 10]
    xpjk.loc[3] = [15, 12, 8]
    xpjk.loc[4] = [5, 7, 3]

    aggregates = [xipp, xpjp, xppk, xijp, xpjk]
    dimensions = [['dma'], ['size'], ['age'], ['dma', 'size'], ['size', 'age']]

    IPF = ipfn.ipfn(df, aggregates, dimensions)
    df = IPF.iteration()

    print(df)
    print(df.groupby('size')['total'].sum(), xpjp)

Added notes:
------------

To call the algorithm in a program, execute::

    from ipfn import ipfn
