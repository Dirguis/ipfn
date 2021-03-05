from __future__ import print_function
from context import ipfn
import numpy as np
import pandas as pd
import pytest


class TestIpfn:

    def test_bad_verbose(self):
        m = np.array([[8., 4., 6., 7.], [3., 6., 5., 2.], [9., 11., 3., 1.]], )
        xip = np.array([20., 18., 22.])
        xpj = np.array([18., 16., 12., 14.])
        aggregates = [xip, xpj]
        dimensions = [[0], [1]]

        with pytest.raises(ValueError):
            IPF = ipfn.ipfn(m, aggregates, dimensions, convergence_rate=1e-5, verbose=4)

    def test_bad_types(self):
        m = [[8., 4., 6., 7.], [3., 6., 5., 2.], [9., 11., 3., 1.]] # not a np.array
        xip = np.array([20., 18., 22.])
        xpj = np.array([18., 16., 12., 14.])
        aggregates = [xip, xpj]
        dimensions = [[0], [1]]

        IPF = ipfn.ipfn(m, aggregates, dimensions, convergence_rate=1e-5)
        with pytest.raises(ValueError):
            m = IPF.iteration()

    def test_numpy_2D(self):
        m = np.array([[8., 4., 6., 7.], [3., 6., 5., 2.], [9., 11., 3., 1.]], )
        xip = np.array([20., 18., 22.])
        xpj = np.array([18., 16., 12., 14.])
        aggregates = [xip, xpj]
        dimensions = [[0], [1]]

        IPF = ipfn.ipfn(m, aggregates, dimensions, convergence_rate=1e-5)
        m = IPF.iteration()

        marginals1D = [xip, xpj]
        m_inc = 0
        for marginal in marginals1D:
            nb_dim = marginal.shape[0]
            for dim in range(nb_dim):
                if m_inc == 0:
                    ipfn_number = np.sum(m[dim, :])
                if m_inc == 1:
                    ipfn_number = np.sum(m[:, dim])
                truth_number = marginal[dim]
                assert round(ipfn_number, 2) == round(truth_number, 2)
            m_inc += 1

    def test_numpy_3D(self):
        m = np.zeros((2, 4, 3))
        m[0, 0, 0] = 1
        m[0, 0, 1] = 2
        m[0, 0, 2] = 1
        m[0, 1, 0] = 3
        m[0, 1, 1] = 5
        m[0, 1, 2] = 5
        m[0, 2, 0] = 6
        m[0, 2, 1] = 2
        m[0, 2, 2] = 2
        m[0, 3, 0] = 1
        m[0, 3, 1] = 7
        m[0, 3, 2] = 2
        m[1, 0, 0] = 5
        m[1, 0, 1] = 4
        m[1, 0, 2] = 2
        m[1, 1, 0] = 5
        m[1, 1, 1] = 5
        m[1, 1, 2] = 5
        m[1, 2, 0] = 3
        m[1, 2, 1] = 8
        m[1, 2, 2] = 7
        m[1, 3, 0] = 2
        m[1, 3, 1] = 7
        m[1, 3, 2] = 6

        xipp = np.array([52, 48])
        xpjp = np.array([20, 30, 35, 15])
        xppk = np.array([35, 40, 25])
        xijp = np.array([[9, 17, 19, 7], [11, 13, 16, 8]])
        xpjk = np.array([[7, 9, 4], [8, 12, 10], [15, 12, 8], [5, 7, 3]])

        aggregates = [xipp, xpjp, xppk, xijp, xpjk]
        dimensions = [[0], [1], [2], [0, 1], [1, 2]]

        IPF = ipfn.ipfn(m, aggregates, dimensions, convergence_rate=0.0001)
        m = IPF.iteration()

        marginals1D = [xipp, xpjp, xppk]
        m_inc = 0
        for marginal in marginals1D:
            nb_dim = marginal.shape[0]
            for dim in range(nb_dim):
                if m_inc == 0:
                    ipfn_number = np.sum(m[dim, :, :])
                if m_inc == 1:
                    ipfn_number = np.sum(m[:, dim, :])
                if m_inc == 2:
                    ipfn_number = np.sum(m[:, :, dim])
                truth_number = marginal[dim]
                assert round(ipfn_number, 2) == round(truth_number, 2)
            m_inc += 1

        marginals2D = [xijp, xpjk]
        m_inc = 0
        for marginal in marginals2D:
            nb_dim1, nb_dim2 = marginal.shape
            for dim1 in range(nb_dim1):
                for dim2 in range(nb_dim2):
                    if m_inc == 0:
                        ipfn_number = np.sum(m[dim1, dim2, :])
                    if m_inc == 1:
                        ipfn_number = np.sum(m[:, dim1, dim2])
                    truth_number = marginal[dim1, dim2]
                    assert round(ipfn_number, 2) == round(truth_number, 2)
            m_inc += 1

    def test_numpy_4D(self):
        m = np.random.rand(2, 5, 4, 3) * 200
        m_new = np.random.rand(2, 5, 4, 3) * 200
        xijkp = np.random.rand(2, 5, 4) * 200
        xpjkl = np.random.rand(5, 4, 3) * 200
        xipkl = np.random.rand(2, 4, 3) * 200
        xijpl = np.random.rand(2, 5, 3) * 200
        xippp = np.random.rand(2) * 200
        xpjpp = np.random.rand(5) * 200
        xppkp = np.random.rand(4) * 200
        xpppl = np.random.rand(3) * 200
        xijpp = np.random.rand(2, 5) * 200
        xpjkp = np.random.rand(5, 4) * 200
        xppkl = np.random.rand(4, 3) * 200
        xippl = np.random.rand(2, 3) * 200

        for i in range(2):
            for j in range(5):
                for k in range(4):
                    xijkp[i, j, k] = m_new[i, j, k, :].sum()
        for j in range(5):
            for k in range(4):
                for l in range(3):
                    xpjkl[j, k, l] = m_new[:, j, k, l].sum()
        for i in range(2):
            for k in range(4):
                for l in range(3):
                    xipkl[i, k, l] = m_new[i, :, k, l].sum()
        for i in range(2):
            for j in range(5):
                for l in range(3):
                    xijpl[i, j, l] = m_new[i, j, :, l].sum()

        for i in range(2):
            xippp[i] = m_new[i, :, :, :].sum()
        for j in range(5):
            xpjpp[j] = m_new[:, j, :, :].sum()
        for k in range(4):
            xppkp[k] = m_new[:, :, k, :].sum()
        for l in range(3):
            xpppl[l] = m_new[:, :, :, l].sum()

        for i in range(2):
            for j in range(5):
                xijpp[i, j] = m_new[i, j, :, :].sum()
        for j in range(5):
            for k in range(4):
                xpjkp[j, k] = m_new[:, j, k, :].sum()
        for k in range(4):
            for l in range(3):
                xppkl[k, l] = m_new[:, :, k, l].sum()
        for i in range(2):
            for l in range(3):
                xippl[i, l] = m_new[i, :, :, l].sum()

        aggregates = [xijkp, xpjkl, xipkl, xijpl, xippp, xpjpp, xppkp, xpppl, xijpp, xpjkp, xppkl, xippl]
        dimensions = [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3], [0], [1], [2], [3], [0, 1], [1, 2], [2, 3], [0, 3]]

        IPF = ipfn.ipfn(m, aggregates, dimensions, convergence_rate=1e-6)
        m = IPF.iteration()

        marginals1D = [xippp, xpjpp, xppkp, xpppl]
        m_inc = 0
        for marginal in marginals1D:
            nb_dim = marginal.shape[0]
            for dim in range(nb_dim):
                if m_inc == 0:
                    ipfn_number = np.sum(m[dim, :, :, :])
                if m_inc == 1:
                    ipfn_number = np.sum(m[:, dim, :, :])
                if m_inc == 2:
                    ipfn_number = np.sum(m[:, :, dim, :])
                if m_inc == 3:
                    ipfn_number = np.sum(m[:, :, :, dim])
                truth_number = marginal[dim]
                assert round(ipfn_number, 2) == round(truth_number, 2)
            m_inc += 1

        marginals2D = [xijpp, xpjkp, xppkl, xippl]
        m_inc = 0
        for marginal in marginals2D:
            nb_dim1, nb_dim2 = marginal.shape
            for dim1 in range(nb_dim1):
                for dim2 in range(nb_dim2):
                    if m_inc == 0:
                        ipfn_number = np.sum(m[dim1, dim2, :, :])
                    if m_inc == 1:
                        ipfn_number = np.sum(m[:, dim1, dim2, :])
                    if m_inc == 2:
                        ipfn_number = np.sum(m[:, :, dim1, dim2])
                    if m_inc == 3:
                        ipfn_number = np.sum(m[dim1, :, :, dim2])
                    truth_number = marginal[dim1, dim2]
                    assert round(ipfn_number, 2) == round(truth_number, 2)
            m_inc += 1

    def test_pandas_3D(self):
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

        aggregates = [xipp, xpjp, xppk, xijp, xpjk]
        dimensions = [['dma'], ['size'], ['age'], ['dma', 'size'], ['size', 'age']]

        IPF = ipfn.ipfn(df, aggregates, dimensions, convergence_rate=1e-5)
        df = IPF.iteration()

        marginals1D = [(xipp, ['dma']), (xpjp, ['size']), (xppk, 'age')]
        m_inc = 0
        for marginal, vertical in marginals1D:
            features = marginal.index.tolist()
            for feature in features:
                assert round(df.groupby(vertical)['total'].sum().loc[feature], 2) == round(marginal.loc[feature], 2)
            m_inc += 1

        marginals2D = [(xijp, ['dma', 'size']), (xpjk, ['size', 'age'])]
        m_inc = 0
        for marginal, vertical in marginals2D:
            features = marginal.index.tolist()
            for feature in features:
                assert round(df.groupby(vertical)['total'].sum().loc[feature], 2) == round(marginal.loc[feature], 2)
            m_inc += 1
