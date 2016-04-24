#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
from itertools import product
import warnings
warnings.filterwarnings('ignore')

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

def ipfn_np(m, aggregates, dimensions):
    steps = len(aggregates)
    dim = len(m.shape)
    product_elem = []
    tables = [m]
    for inc in range(steps-1):
        tables.append(np.array(np.zeros(m.shape)))

    for inc in range(steps):
        if inc == (steps-1):
            table_update  = m
            table_current = tables[inc]
        else:
            table_update  = tables[inc+1]
            table_current = tables[inc]
        for dimension in dimensions[inc]:
            product_elem.append(range(m.shape[dimension]))
        for item in product(*product_elem):
            idx = index_axis_elem(dim, dimensions[inc], item)
            table_current_slice = table_current[idx]
            mijk = table_current_slice.sum()
            xijk  = aggregates[inc]
            xijk = xijk[item]
            if mijk == 0:
                # table_current_slice += 1e-5
                table_update[idx] = table_current_slice*1.0*xijk
            else:
                table_update[idx] = table_current_slice*1.0*xijk/mijk
            # For debug purposes
            # if np.isnan(table_update).any():
            #     print idx
            #     sys.exit(0)
        product_elem = []
    return m



def ipfn_df(df, aggregates, dimensions):

    steps = len(aggregates)
    tables = [df]
    for inc in range(steps-1):
            tables.append(df.copy())

    inc=0
    for features in dimensions:
        if inc == (steps-1):
            table_update  = df
            table_current = tables[inc]
        else:
            table_update  = tables[inc+1]
            table_current = tables[inc]

        tmp  = table_current.groupby(features)['total'].sum()
        xijk = aggregates[inc]

        feat_l = []
        for feature in features:
            feat_l.append(np.unique(table_current[feature]))
        table_update.set_index(features, inplace=True)
        table_current.set_index(features, inplace=True)
        # table_update.sortlevel(inplace=True,sort_remaining=True)
        # table_current.sortlevel(inplace=True,sort_remaining=True)

        for feature in product(*feat_l):

            den = tmp.loc[feature]
            if den.sum() == 0:
                table_update.loc[feature, 'total'] =\
                table_current.loc[feature, 'total']*\
                xijk.loc[feature]
            else:
                table_update.loc[feature, 'total'] =\
                table_current.loc[feature, 'total']*\
                xijk.loc[feature]/den

        table_update.reset_index(inplace=True)
        table_current.reset_index(inplace=True)
        inc+=1
        feat_l = []
    return df

if __name__ == '__main__':

    # Example 1, 2D using ipfn_np, link: http://www.real-statistics.com/matrices-and-iterative-procedures/iterative-proportional-fitting-procedure-ipfp/
    # m = np.array([[8., 4., 6., 7.], [3., 6., 5., 2.], [9., 11., 3., 1.]], )
    # xip = np.array([20., 18., 22.])
    # xpj = np.array([18., 16., 12., 14.])
    # aggregates = [xip, xpj]
    # dimensions = [[0], [1]]
    #
    # for inc in range(10):
    #     m = ipfn_np(m, aggregates, dimensions)
    # print m
    # print m[0,:].sum(), xip[0]


    # Example 2, 3D using ipfn_np, link: http://www.demog.berkeley.edu/~eddieh/IPFDescription/AKDOLWDIPFTHREED.pdf
    # There is a link to a excel file with the example if interested
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

    xipp = np.array([52, 48])
    xpjp = np.array([20, 30, 35, 15])
    xppk = np.array([35, 40, 25])
    xijp = np.array([[9, 17, 19, 7], [11, 13, 16, 8]])
    # xijp = xijp.T
    xpjk = np.array([[7, 9, 4], [8, 12, 10], [15, 12, 8], [5, 7, 3]])
    aggregates = [xipp, xpjp, xppk, xijp, xpjk]
    dimensions = [[0], [1], [2], [0, 1], [1, 2]]

    for inc in range(10):
        m = ipfn_np(m, aggregates, dimensions)
    print xijp[0,0]
    print m[0, 0, :].sum()



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
    # for inc in range(30):
    #     m = ipfn_np(m, aggregates, dimensions)
    #
    # print m
    # print xpjkl[2,1,2], m[:,2,1,2].sum()
    # print xpjpp[1], m[:,1,:,:].sum()
    # print xppkl[0, 2], m[:,:,0,2].sum()




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
    # for inc in range(10):
    #     df = ipfn_df(df, [xip, xpj], [['dma'], ['size']])
    #
    # print df
    # print df.groupby('dma')['total'].sum(), xip



    # Example 3D with ipfn_df
    # m      = np.array([1., 2., 1., 3., 5., 5., 6., 2., 2., 1., 7., 2.,
    #                5., 4., 2., 5., 5., 5., 3., 8., 7., 2., 7., 6.], )
    # dma_l  = [501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501, 501,
    #           502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502, 502]
    # size_l = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
    #           1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    #
    # age_l  = ['20-25','30-35','40-45',
    #           '20-25','30-35','40-45',
    #           '20-25','30-35','40-45',
    #           '20-25','30-35','40-45',
    #           '20-25','30-35','40-45',
    #           '20-25','30-35','40-45',
    #           '20-25','30-35','40-45',
    #           '20-25','30-35','40-45']
    #
    # df = pd.DataFrame()
    # df['dma'] = dma_l
    # df['size'] = size_l
    # df['age'] = age_l
    # df['total'] = m
    #
    # xipp = df.groupby('dma')['total'].sum()
    # xpjp = df.groupby('size')['total'].sum()
    # xppk = df.groupby('age')['total'].sum()
    # xijp = df.groupby(['dma', 'size'])['total'].sum()
    # xpjk = df.groupby(['size', 'age'])['total'].sum()
    # # xppk = df.groupby('age')['total'].sum()
    #
    # xipp.loc[501] = 52
    # xipp.loc[502] = 48
    #
    # xpjp.loc[1] = 20
    # xpjp.loc[2] = 30
    # xpjp.loc[3] = 35
    # xpjp.loc[4] = 15
    #
    # xppk.loc['20-25'] = 35
    # xppk.loc['30-35'] = 40
    # xppk.loc['40-45'] = 25
    #
    # xijp.loc[501] = [9, 17, 19, 7]
    # xijp.loc[502] = [11, 13, 16, 8]
    #
    # xpjk.loc[1] = [7, 9, 4]
    # xpjk.loc[2] = [8, 12, 10]
    # xpjk.loc[3] = [15, 12, 8]
    # xpjk.loc[4] = [5, 7, 3]
    #
    # for inc in range(10):
    #     df = ipfn_df(df, [xipp, xpjp, xppk, xijp, xpjk],
    #             [['dma'], ['size'], ['age'], ['dma', 'size'], ['size', 'age']])
    #
    # print df
    # print df.groupby('size')['total'].sum(), xpjp
