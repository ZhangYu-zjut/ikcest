'''
generate adjacent matrix
adjacent matrix "adjmat_*.txt" is an important input file for the model,
adjmat reflects the relationship between zones, including two parts:
(1)inner city: voronoi matrix (if adjacent in spatial), "vonoroi_mat.txt"
(2)intra city: hubs matrix (if relate in transfer), consider lock in the city
'''

import pandas as pd  # 1.0.3
import numpy as np  # 1.18.2
import math
import os
from scipy.sparse import csc_matrix  # 1.4.1

def make_hubs_matrix(src_path, city_dict, top_rate=0.05, close_citys=[]):
    # intra city: city zone-hubs relationship based on transfer.csv
    zone_all = pd.read_csv('%s/zone_all.csv' % src_path)
    hubs_all = zone_all.copy(deep=True)
    city_list = city_dict.keys()

    for city_name in city_list:
        # find the zone-hubs list of every city based on the zone transfer aggregated data
        tsfpair = pd.read_csv('%s/transfer_inout_%s.csv' % (src_path, city_name))
        zone_num = city_dict[city_name]
        hub_num = math.ceil(zone_num * top_rate)
        hubs = tsfpair.sort_values('tsfpair', ascending=False).head(hub_num).zoneid.tolist()
        hubs_all.loc[(hubs_all.city == city_name)
                     & (hubs_all.zoneid.isin(hubs)),
                     'hub'] = 1

    # screen all the city hubs zone
    hubs_all = hubs_all[hubs_all.hub == 1]
    hubs_all = hubs_all.merge(hubs_all, on='hub')
    hubs_all = hubs_all[hubs_all.city_x != hubs_all.city_y]
    hubs_all = hubs_all[(~hubs_all.city_x.isin(close_citys)) & (~hubs_all.city_y.isin(close_citys))]

    # generate the sparse matrix of intra city hub relationship
    hubs_x = hubs_all.zoneid1_x.values  # one hub in one city
    hubs_y = hubs_all.zoneid1_y.values  # one hub in another city
    hubs_v = np.ones(len(hubs_x))  # mask=1: it is a hub
    hubs_mat = csc_matrix((hubs_v, (hubs_x, hubs_y)),
                          shape=(len(zone_all), len(zone_all)))
    hubs_mat = hubs_mat.toarray()

    return pd.DataFrame(hubs_mat)


def main():
    src_path = 'src4'
    city_list = list('ABCDEFGHIJK')
    city_dict = dict(zip(city_list, [118, 30, 135, 75, 34, 331, 38, 53, 33, 8, 48]))

    # vonoroi matrix: inner city--the spatial adjacent between two zones in one city
    vonoroi_mat=pd.read_csv('%s/vonoroi_mat.txt'%src_path,header=None)

    # hubs_mask_matrix: intra city--the mobility relationship between two zones in two cities
    hubs_mask_Y055 = make_hubs_matrix(src_path, city_dict, 0.30, list('AF'))
    adjmat_Y055 = (vonoroi_mat + hubs_mask_Y055).astype(int)
    adjmat_Y055.to_csv('%s/adjmat_Y055_voronoi0.30HubNoAF.txt' % src_path, index=False, header=False)

    hubs_mask_Y064 = make_hubs_matrix(src_path, city_dict, 0.40, list('AF'))
    adjmat_Y064 = (vonoroi_mat + hubs_mask_Y064).astype(int)
    adjmat_Y064.to_csv('%s/adjmat_Y064_voronoi0.40HubNoAF.txt' % src_path, index=False, header=False)

    hubs_mask_Y075 = make_hubs_matrix(src_path, city_dict, 0.20, list('AF'))
    adjmat_Y075 = (vonoroi_mat + hubs_mask_Y075).apply(lambda x: x / sum(x), axis=1)
    adjmat_Y075.to_csv('%s/adjmat_Y075_voronoi0.20HubNoAF_rowNorm.txt' % src_path, index=False, header=False)

    hubs_mask_Y104 = make_hubs_matrix(src_path, city_dict, 0.05, list('ACFIJK'))
    adjmat_Y104 = (vonoroi_mat + hubs_mask_Y104).astype(int)
    adjmat_Y104.to_csv('%s/adjmat_Y104_voronoi0.05HubNoACFIJK.txt' % src_path, index=False, header=False)

    hubs_mask_Y159 = make_hubs_matrix(src_path, city_dict, 0.90, list('ACFIJK'))
    adjmat_Y159 = (vonoroi_mat + hubs_mask_Y159).astype(int)
    adjmat_Y159.to_csv('%s/adjmat_Y159_voronoi0.90HubNoACFIJK.txt' % src_path, index=False, header=False)


if __name__ == '__main__':
    main()
