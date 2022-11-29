from __future__ import print_function
import numpy as np
import pandas as pd
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
        if verbose not in [0, 1, 2]:
            raise ValueError(f"wrong verbose input, must be either 0, 1 or 2 but got {verbose}")
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

        # Check that the inputs are numpy arrays of floats
        inc = 0
        for aggregate in aggregates:
            if not isinstance(aggregate, np.ndarray):
                aggregate = np.array(aggregate).astype(float)
                aggregates[inc] = aggregate
            elif aggregate.dtype not in [float, float]:
                aggregate = aggregate.astype(float)
                aggregates[inc] = aggregate
            inc += 1
        if not isinstance(m, np.ndarray):
            m = np.array(m)
        elif m.dtype not in [float, float]:
            m = m.astype(float)

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
                table_current = tables[inc].copy()
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
                # print("Current vs original", abs(m_ijk/ori_ijk - 1))
                if abs(m_ijk / ori_ijk - 1) > max_conv:
                    max_conv = abs(m_ijk / ori_ijk - 1)

            product_elem = []

        return m, max_conv

    def ipfn_df(self, df, aggregates, dimensions, weight_col='total'):
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

        steps = len(aggregates)
        tables = [df]
        for inc in range(steps - 1):
            tables.append(df.copy())
        original = df.copy()

        # Calculate the new weights for each dimension
        inc = 0
        for features in dimensions:
            if inc == (steps - 1):
                table_update = df
                table_current = tables[inc].copy()
            else:
                table_update = tables[inc + 1]
                table_current = tables[inc]

            tmp = table_current.groupby(features)[weight_col].sum()
            xijk = aggregates[inc]

            feat_l = []
            for feature in features:
                feat_l.append(np.unique(table_current[feature]))
            table_update.set_index(features, inplace=True)
            table_current.set_index(features, inplace=True)

            multi_index_flag = isinstance(table_update.index, pd.MultiIndex)
            if multi_index_flag:
                if not table_update.index.is_monotonic_increasing:
                    table_update.sort_index(inplace=True)
                if not table_current.index.is_monotonic_increasing:
                    table_current.sort_index(inplace=True)

            for feature in product(*feat_l):
                den = tmp.loc[feature]
                # calculate new weight for this iteration

                if not multi_index_flag:
                    msk = table_update.index == feature[0]
                else:
                    msk = feature

                if den == 0:
                    table_update.loc[msk, weight_col] =\
                        table_current.loc[feature, weight_col] *\
                        xijk.loc[feature]
                else:
                    table_update.loc[msk, weight_col] = \
                        table_current.loc[feature, weight_col].astype(float) * \
                        xijk.loc[feature] / den

            table_update.reset_index(inplace=True)
            table_current.reset_index(inplace=True)
            inc += 1
            feat_l = []

        # Calculate the max convergence rate
        max_conv = 0
        inc = 0
        for features in dimensions:
            tmp = table_update.groupby(features)[weight_col].sum()
            ori_ijk = aggregates[inc]
            temp_conv = max(abs(tmp / ori_ijk - 1))
            if temp_conv > max_conv:
                max_conv = temp_conv
            inc += 1

        return table_update, max_conv

    def iteration(self):
        """
        Runs the ipfn algorithm. Automatically detects of working with numpy ndarray or pandas dataframes.
        """
        old_conv = np.inf
        conv_list = []
        converged = 1
        m = self.original

        # Prepare input data
        if isinstance(self.original, pd.DataFrame):
            ipfn_method = self.ipfn_df
        elif isinstance(self.original, np.ndarray):
            ipfn_method = self.ipfn_np
            self.original = self.original.astype('float64')
        else:
            raise ValueError(f"Data input instance not recognized. The input matrix is not a numpy array or pandas DataFrame")

        # Run iterations
        for i in range(self.max_itr):
            m, conv = ipfn_method(m, self.aggregates, self.dimensions, self.weight_col)
            conv_list.append(conv)
            if conv <= self.conv_rate:
                if self.verbose > 1:
                    print("ipfn converged: convergence_rate below threshold")
                break
            if abs(conv - old_conv) <= self.rate_tolerance:
                print("ipfn converged: convergence_rate not updating or below rate_tolerance")
                break
            old_conv = conv
        else:
            print("Maximum iterations reached")
            converged = 0

        # Handle the verbose
        if self.verbose == 0:
            return m
        elif self.verbose == 1:
            return m, converged
        elif self.verbose == 2:
            return m, converged, pd.DataFrame({'iteration': range(1, i+2), 'conv': conv_list}).set_index('iteration')
        else:
            raise ValueError(f"wrong verbose input, must be either 0, 1 or 2 but got {self.verbose}")
