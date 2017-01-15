#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
from itertools import product
import copy


class ipfn(object):

    def __init__(self, original, aggregates, dimensions, convergence_rate=0.0001, max_iteration=500, verbose=0):
        """
        Initialize the ipfn class
        original: numpy darray matrix or dataframe to perform the ipfn on.
        aggregates: list of numpy array or darray or pandas dataframe/series. The aggregates are the same as the marginals. They are the target values that we want along one or several axis when aggregating along one or several axes.
        dimensions: list of lists with integers if working with numpy objects, or column names if working with pandas objects. Preserved dimensions along which we sum to get the corresponding aggregates.
        convergence_rate: if there are many aggregates/marginal, it could be useful to loosen the convergence criterion.
        max_iteration: Integer. Maximum number of iterations allowed.
        verbose: interger 0 or 1. Returns 1 if the ipfn successfully converged, 0 otherwise.

        For examples, please open the ipfn script or look for help on functions ipfn_np and ipfn_df
        """
        self.original = original
        self.aggregates = aggregates
        self.dimensions = dimensions
        self.conv_rate = convergence_rate
        self.max_itr = max_iteration
        self.verbose = verbose

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

    def ipfn_np(self, m, aggregates, dimensions):
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
        for inc in range(steps-1):
            tables.append(np.array(np.zeros(m.shape)))
        original = copy.copy(m)

        # Calculate the new weights for each dimension
        for inc in range(steps):
            if inc == (steps-1):
                table_update = m
                table_current = tables[inc]
            else:
                table_update = tables[inc+1]
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
                    table_update[idx] = table_current_slice*1.0*xijk/mijk
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
                ori_slice = original[idx]
                ori_ijk = ori_slice.sum()
                m_slice = m[idx]
                m_ijk = m_slice.sum()
                # print('Current vs original', abs(m_ijk/ori_ijk - 1))
                if abs(m_ijk/ori_ijk - 1) > max_conv:
                    max_conv = abs(m_ijk/ori_ijk - 1)

            product_elem = []

        return m, max_conv

    # TODO: Should we still keep this function as we no longer using dataframe version? (Evelyn)
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

        steps = len(aggregates)
        tables = [df]
        for inc in range(steps-1):
            tables.append(df.copy())
        original = df.copy()

        # Calculate the new weights for each dimension
        inc = 0
        for features in dimensions:
            if inc == (steps-1):
                table_update = df
                table_current = tables[inc]
            else:
                table_update = tables[inc+1]
                table_current = tables[inc]

            tmp = table_current.groupby(features)['total'].sum()
            xijk = aggregates[inc]

            feat_l = []
            for feature in features:
                feat_l.append(np.unique(table_current[feature]))
            table_update.set_index(features, inplace=True)
            table_current.set_index(features, inplace=True)

            for feature in product(*feat_l):

                den = tmp.loc[feature]
                if den == 0:
                    table_update.loc[feature, 'total'] =\
                        table_current.loc[feature, 'total'] *\
                        xijk.loc[feature]
                else:
                    table_update.loc[feature, 'total'] =\
                        table_current.loc[feature, 'total'] *\
                        xijk.loc[feature]/den

            table_update.reset_index(inplace=True)
            table_current.reset_index(inplace=True)
            inc += 1
            feat_l = []

        # Calculate the max convergence rate
        max_conv = 0
        inc = 0
        for features in dimensions:
            tmp = df.groupby(features)['total'].sum()
            ori_ijk = original.groupby(features)['total'].sum()
            temp_conv = max(abs(tmp/ori_ijk - 1))
            if temp_conv > max_conv:
                max_conv = temp_conv
            inc += 1

        return df, max_conv

    def iteration(self):
        """
        Runs the ipfn algorithm. Automatically detects of working with numpy ndarray or pandas dataframes.
        """

        i = 0
        conv = self.conv_rate * 100
        m = self.original

        # If the original data input is in pandas DataFrame format
        if isinstance(self.original, pd.DataFrame):
            while i <= self.max_itr and conv > self.conv_rate:
                m, conv = self.ipfn_df(m, self.aggregates, self.dimensions)
                i += 1
                # print(i, conv)
        # If the original data input is in numpy format
        elif isinstance(self.original, np.ndarray):
            self.original = self.original.astype('float64')
            while i <= self.max_itr and conv > self.conv_rate:
                m, conv = self.ipfn_np(m, self.aggregates, self.dimensions)
                i += 1
                # print(i, conv)
        converged = 1
        if i <= self.max_itr:
            print('ipfn converged')
        else:
            print('Maximum iterations reached')
            converged = 0

        # Handle the verbose
        if self.verbose == 0:
            return m
        elif self.verbose == 1:
            return m, converged
        else:
            print('wrong verbose input, return None')
            sys.exit(0)

if __name__ == '__main__':

    # Example 1, 2D using ipfn_np,
    # link: http://www.real-statistics.com/matrices-and-iterative-procedures/iterative-proportional-fitting-procedure-ipfp/
    # m = np.array([[8., 4., 6., 7.], [3., 6., 5., 2.], [9., 11., 3., 1.]], )
    # xip = np.array([20., 18., 22.])
    # xpj = np.array([18., 16., 12., 14.])
    # aggregates = [xip, xpj]
    # dimensions = [[0], [1]]
    #
    # IPF = ipfn(m, aggregates, dimensions)
    # m = IPF.iteration()
    #
    # print(m)
    # print(m[0,:].sum(), xip[0])

    # Example 2, 3D using ipfn_np, link: http://www.demog.berkeley.edu/~eddieh/IPFDescription/AKDOLWDIPFTHREED.pdf
    # There is a link to a excel file with the example if interested
    # m = np.zeros((2,4,3))
    # m[0,0,0] = 1
    # m[0,0,1] = 2
    # m[0,0,2] = 1
    # m[0,1,0] = 3
    # m[0,1,1] = 5
    # m[0,1,2] = 5
    # m[0,2,0] = 6
    # m[0,2,1] = 2
    # m[0,2,2] = 2
    # m[0,3,0] = 1
    # m[0,3,1] = 7
    # m[0,3,2] = 2
    #
    # m[1,0,0] = 5
    # m[1,0,1] = 4
    # m[1,0,2] = 2
    # m[1,1,0] = 5
    # m[1,1,1] = 5
    # m[1,1,2] = 5
    # m[1,2,0] = 3
    # m[1,2,1] = 8
    # m[1,2,2] = 7
    # m[1,3,0] = 2
    # m[1,3,1] = 7
    # m[1,3,2] = 6
    #
    # xipp = np.array([52, 48])
    # xpjp = np.array([20, 30, 35, 15])
    # xppk = np.array([35, 40, 25])
    # xijp = np.array([[9, 17, 19, 7], [11, 13, 16, 8]])
    # # xijp = xijp.T
    # xpjk = np.array([[7, 9, 4], [8, 12, 10], [15, 12, 8], [5, 7, 3]])
    # aggregates = [xipp, xpjp, xppk, xijp, xpjk]
    # dimensions = [[0], [1], [2], [0, 1], [1, 2]]
    #
    # IPF = ipfn(m, aggregates, dimensions)
    # m = IPF.iteration()
    # print(m)
    # print(m[0, 0, :].sum())

    # Example 3, 4D using ipfn_np, link: http://www.demog.berkeley.edu/~eddieh/IPFDescription/AKDOLWDIPFFOURD.pdf
    # made up example

    # m = np.random.rand(2,5,4,3)*200
    # m_new = np.random.rand(2,5,4,3)*200
    # xijkp = np.random.rand(2,5,4)*200
    # xpjkl = np.random.rand(5,4,3)*200
    # xipkl = np.random.rand(2,4,3)*200
    # xijpl = np.random.rand(2,5,3)*200
    # xippp = np.random.rand(2)*200
    # xpjpp = np.random.rand(5)*200
    # xppkp = np.random.rand(4)*200
    # xpppl = np.random.rand(3)*200
    # xijpp = np.random.rand(2,5)*200
    # xpjkp = np.random.rand(5,4)*200
    # xppkl = np.random.rand(4,3)*200
    # xippl = np.random.rand(2,3)*200
    #
    # for i in range(2):
    #     for j in range(5):
    #         for k in range(4):
    #             xijkp[i,j,k] = m_new[i,j,k,:].sum()
    # for j in range(5):
    #     for k in range(4):
    #         for l in range(3):
    #             xpjkl[j,k,l] = m_new[:,j,k,l].sum()
    # for i in range(2):
    #     for k in range(4):
    #         for l in range(3):
    #             xipkl[i,k,l] = m_new[i,:,k,l].sum()
    # for i in range(2):
    #     for j in range(5):
    #         for l in range(3):
    #             xijpl[i,j,l] = m_new[i,j,:,l].sum()
    #
    # for i in range(2):
    #     xippp[i] = m_new[i,:,:,:].sum()
    # for j in range(5):
    #     xpjpp[j] = m_new[:,j,:,:].sum()
    # for k in range(4):
    #     xppkp[k] = m_new[:,:,k,:].sum()
    # for l in range(3):
    #     xpppl[l] = m_new[:,:,:,l].sum()
    #
    # for i in range(2):
    #     for j in range(5):
    #         xijpp[i,j] = m_new[i,j,:,:].sum()
    # for j in range(5):
    #     for k in range(4):
    #         xpjkp[j,k] = m_new[:,j,k,:].sum()
    # for k in range(4):
    #     for l in range(3):
    #         xppkl[k,l] = m_new[:,:,k,l].sum()
    # for i in range(2):
    #     for l in range(3):
    #         xippl[i,l] = m_new[i,:,:,l].sum()
    #
    # aggregates = [xijkp, xpjkl, xipkl, xijpl, xippp, xpjpp, xppkp, xpppl,
    # xijpp, xpjkp, xppkl, xippl]
    # dimensions = [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3], [0], [1], [2], [3],
    # [0, 1], [1, 2], [2, 3], [0, 3]]
    #
    # IPF = ipfn(m, aggregates, dimensions)
    # m = IPF.iteration()
    #
    # print(m)
    # print(xpjkl[2,1,2], m[:,2,1,2].sum())
    # print(xpjpp[1], m[:,1,:,:].sum())
    # print(xppkl[0, 2], m[:,:,0,2].sum())

    # Example 2D with ipfn_df
    # m      = np.array([8., 4., 6., 7., 3., 6., 5., 2., 9., 11., 3., 1.], )
    # dma_l  = [501, 501, 501, 501, 502, 502, 502, 502, 505, 505, 505, 505]
    # size_l = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    #
    # df = pd.DataFrame()
    # df['dma'] = dma_l
    # df['size'] = size_l
    # df['total'] = m
    #
    # xip = df.groupby('dma')['total'].sum()
    # xpj = df.groupby('size')['total'].sum()
    # # df = df.groupby(['dma', 'size']).sum()
    #
    # xip.loc[501] = 20
    # xip.loc[502] = 18
    # xip.loc[505] = 22
    #
    # xpj.loc[1] = 18
    # xpj.loc[2] = 16
    # xpj.loc[3] = 12
    # xpj.loc[4] = 14
    #
    # ipfn_df = ipfn(df, [xipp, xpjp, xppk, xijp, xpjk],
    #         [['dma'], ['size'], ['age'], ['dma', 'size'], ['size', 'age']])
    # df = ipfn_df.iteration()
    #
    # print(df)
    # print(df.groupby('dma')['total'].sum(), xip)

    # # Example 3D with ipfn_df
    m = np.array([1., 2., 1., 3., 5., 5., 6., 2., 2., 1., 7., 2.,
                  5., 4., 2., 5., 5., 5., 3., 8., 7., 2., 7., 6.], )
    dma_l = [501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501,
             502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502]
    size_l = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
              1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]

    age_l = ['20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45',
             '20-25', '30-35', '40-45']

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

    ipfn_df = ipfn(df, [xipp, xpjp, xppk, xijp, xpjk],
                   [['dma'], ['size'], ['age'], ['dma', 'size'], ['size', 'age']])
    df = ipfn_df.iteration()

    print(df)
    print(df.groupby('size')['total'].sum(), xpjp)
