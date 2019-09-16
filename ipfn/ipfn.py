#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
from itertools import product
import copy


class ipfn(object):

    def __init__(self, original, aggregates, dimensions, weight_col='total',
                 convergence_rate=1e-5, max_iteration=500, verbose=0, rate_tolerance=1e-8):
        """
        Initialize the ipfn class

        original: numpy darray matrix or dataframe to perform the ipfn on.

        aggregates: list of numpy array or darray or pandas dataframe/series. The aggregates are the same as the marginals.
        They are the target values that we want along one or several axis when aggregating along one or several axes.

        dimensions: list of lists with integers if working with numpy objects, or column names if working with pandas objects.
        Preserved dimensions along which we sum to get the corresponding aggregates.

        convergence_rate: if there are many aggregates/marginal, it could be useful to loosen the convergence criterion.

        max_iteration: Integer. Maximum number of iterations allowed.

        verbose: integer 0, 1 or 2. Each case number includes the outputs of the previous case numbers.
        0: Updated matrix returned.
        1: Flag with the output status (0 for failure and 1 for success).
        2: dataframe with iteration numbers and convergence rate information at all steps.

        rate_tolerance: float value. If above 0.0, like 0.001, the algorithm will stop once the difference between the conv_rate variable of 2 consecutive iterations is below that specified value

        For examples, please open the ipfn script or look for help on functions ipfn_np and ipfn_df
        """
        self.original = original
        self.aggregates = aggregates
        self.dimensions = dimensions
        self.weight_col = weight_col
        self.conv_rate = convergence_rate
        self.max_itr = max_iteration
        self.verbose = verbose
        self.rate_tolerance = rate_tolerance

    @staticmethod
    def index_axis_elem(dims, axes, elems):
        inc_axis = 0
        idx = ()
        for dim in range(dims):
            if (inc_axis < len(axes)):
                if (dim == axes[inc_axis]):
                    idx += (elems[inc_axis],)
                    inc_axis += 1
                else:
                    idx += (np.s_[:],)
        return idx

    def ipfn_np(self, m, aggregates, dimensions, weight_col='total'):
        """
        Runs the ipfn method from a matrix m, aggregates/marginals and the dimension(s) preserved.
        For example:
        from ipfn import ipfn
        import numpy as np
        m = np.array([[8., 4., 6., 7.], [3., 6., 5., 2.], [9., 11., 3., 1.]], )
        xip = np.array([20., 18., 22.])
        xpj = np.array([18., 16., 12., 14.])
        aggregates = [xip, xpj]
        dimensions = [[0], [1]]

        IPF = ipfn(m, aggregates, dimensions)
        m = IPF.iteration()
        """
        steps = len(aggregates)
        dim = len(m.shape)
        product_elem = []
        tables = [m]
        # TODO: do we need to persist all these dataframe? Or maybe we just need to persist the table_update and table_current
        # and then update the table_current to the table_update to the latest we have. And create an empty zero dataframe for table_update (Evelyn)
        for inc in range(steps - 1):
            tables.append(np.array(np.zeros(m.shape)))
        original = copy.copy(m)

        # Calculate the new weights for each dimension
        for inc in range(steps):
            if inc == (steps - 1):
                table_update = m
                table_current = tables[inc]
            else:
                table_update = tables[inc + 1]
                table_current = tables[inc]
            for dimension in dimensions[inc]:
                product_elem.append(range(m.shape[dimension]))
            for item in product(*product_elem):
                idx = self.index_axis_elem(dim, dimensions[inc], item)
                table_current_slice = table_current[idx]
                mijk = table_current_slice.sum()
                # TODO: Directly put it as xijk = aggregates[inc][item] (Evelyn)
                xijk = aggregates[inc]
                xijk = xijk[item]
                if mijk == 0:
                    # table_current_slice += 1e-5
                    # TODO: Basically, this part would remain 0 as always right? Cause if the sum of the slice is zero, then we only have zeros in this slice.
                    # TODO: you could put it as table_update[idx] = table_current_slice (since multiplication on zero is still zero)
                    table_update[idx] = table_current_slice
                else:
                    # TODO: when inc == steps - 1, this part is also directly updating the dataframe m (Evelyn)
                    # If we are not going to persist every table generated, we could still keep this part to directly update dataframe m
                    table_update[idx] = table_current_slice * 1.0 * xijk / mijk
                # For debug purposes
                # if np.isnan(table_update).any():
                #     print(idx)
                #     sys.exit(0)
            product_elem = []

        # Check the convergence rate for each dimension
        max_conv = 0
        for inc in range(steps):
            # TODO: this part already generated before, we could somehow persist it. But it's not important (Evelyn)
            for dimension in dimensions[inc]:
                product_elem.append(range(m.shape[dimension]))
            for item in product(*product_elem):
                idx = self.index_axis_elem(dim, dimensions[inc], item)
                ori_ijk = aggregates[inc][item]
                m_slice = m[idx]
                m_ijk = m_slice.sum()
                # print('Current vs original', abs(m_ijk/ori_ijk - 1))
                if abs(m_ijk / ori_ijk - 1) > max_conv:
                    max_conv = abs(m_ijk / ori_ijk - 1)

            product_elem = []

        return m, max_conv

    def ipfn_df(self, df, aggregates, dimensions):
        """
        Runs the ipfn method from a dataframe df, aggregates/marginals and the dimension(s) preserved.
        For example:
        from ipfn import ipfn
        import pandas as pd
        age = [30, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50]
        distance = [10,20,30,40,10,20,30,40,10,20,30,40]
        m = [8., 4., 6., 7., 3., 6., 5., 2., 9., 11., 3., 1.]
        df = pd.DataFrame()
        df['age'] = age
        df['distance'] = distance
        df['total'] = m

        xip = df.groupby('age')['total'].sum()
        xip.loc[30] = 20
        xip.loc[40] = 18
        xip.loc[50] = 22
        xpj = df.groupby('distance')['total'].sum()
        xpj.loc[10] = 18
        xpj.loc[20] = 16
        xpj.loc[30] = 12
        xpj.loc[40] = 14
        dimensions = [['age'], ['distance']]
        aggregates = [xip, xpj]

        IPF = ipfn(df, aggregates, dimensions)
        df = IPF.iteration()

        print(df)
        print(df.groupby('age')['total'].sum(), xip)"""

        aggregates = self.aggregates
        dimensions = self.dimensions
        factors = []
        index_names = df.index.names

        for k, d in enumerate(dimensions):
            dfg = df.groupby(level=d).sum()
            f = aggregates[k].div(dfg)
            # Requires pandas >= 0.25
            if len(d) > 1:
                rem_index = [lvl for lvl in index_names if lvl in d]
                df = (df.multiply(f.reorder_levels(rem_index), axis=0)
                        .reorder_levels(index_names))
            else:
                df = df.multiply(f, fill_value=0)

            f = f.sub(1).abs().max()
            factors.append(f)

        # Check for convergence
        max_conv = max(factors)

        return df, max_conv

    def iteration(self):
        """
        Runs the ipfn algorithm. Automatically detects of working with
        numpy ndarray or pandas dataframes.
        """

        i = 0
        conv = self.conv_rate * 100
        m = self.original.copy()

        # If the original data input is in pandas DataFrame format
        if isinstance(self.original, pd.DataFrame):
            # Add index
            indexcols = list(set(x for l in self.dimensions for x in l))
            m.reset_index(inplace=True)
            m.set_index(indexcols, inplace=True)
            # Turn to series
            m = m[self.weight_col]
            while i <= self.max_itr and conv > self.conv_rate:
                m, conv = self.ipfn_df(m, self.aggregates, self.dimensions)
                i += 1
                # print(i, conv)
        # If the original data input is in numpy format
        elif isinstance(self.original, np.ndarray):
            self.original = self.original.astype('float64')
            while i <= self.max_itr and conv > self.conv_rate:
                m, conv = self.ipfn_np(m, self.aggregates, self.dimensions, self.weight_col)
                i += 1
                # print(i, conv)

        converged = True
        if i <= self.max_itr:
            print('ipfn converged')
        else:
            print('Maximum iterations reached')
            converged = False

        # Handle the verbose
        if self.verbose == 0:
            return m
        elif self.verbose == 1:
            return m, converged
        else:
            print('wrong verbose input, return None')
            sys.exit(0)
