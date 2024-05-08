import itertools
import pickle
import pygeohash as pgh
import pygeohash_fast
import os
os.environ['USE_PYGEOS'] = '0'
import networkx as nx
import numba
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import snap, split
import geopandas as gpd
from scipy.sparse import csr_matrix
from itertools import chain
import osmnx as ox


"""
Utility functions
"""
def edge_subsampling(edge_geometry, edge_length, dist_max = 5):
    geom_edge, len_edge = edge_geometry, round(edge_length) 
    if len_edge > dist_max:
        geom_edge = MultiPoint([geom_edge.interpolate(i/len_edge, normalized=True) for i in range(0,len_edge+1,dist_max)])
    elif len_edge ==0:
        geom_edge = MultiPoint(list(geom_edge.coords))
    else:
        geom_edge = geom_edge.boundary
    return(geom_edge)

def interpolation_edge_geometry(G, dist_max = 5):
    interpolated_geom = [edge_subsampling(G.edges[i]['geometry'],G.edges[i]['length'], dist_max = dist_max) for i in G.edges]
    keys = list(G.edges)
    edge2newinterpolatedgeom = dict(zip(keys, interpolated_geom))
    return(edge2newinterpolatedgeom)

def get_geohashes_of_edge(G, show_print=True):
    edges2interpolation = interpolation_edge_geometry(G, dist_max = 5)
    geom_interpolation = pd.DataFrame(edges2interpolation.items(),columns=['edge', 'geometry'])
    geom_interpolation = gpd.GeoDataFrame(geom_interpolation, geometry=geom_interpolation['geometry'])
    geom_interpolation = geom_interpolation.explode('geometry', index_parts=True)
    geom_interpolation["lon"] = geom_interpolation["geometry"].x
    geom_interpolation["lat"] = geom_interpolation["geometry"].y
    geom_interpolation['geohash']   = pygeohash_fast.encode_many(geom_interpolation['lon'].values, geom_interpolation['lat'].values, 8)
    edge_geohash = geom_interpolation.groupby('edge')['geohash'].apply(set).apply(list).to_dict()
    nx.set_edge_attributes(G, edge_geohash, 'geohashes')
    
    if show_print:
        print("Get geohashes of edges done")
    return(G)

def linestring_length(geom_edge):
    coords = np.array(geom_edge.coords)
    length = 1000 *sum(haversine(coords[:-1, 1], coords[:-1, 0], coords[1:, 1], coords[1:, 0]))
    return(length)

def remove_unneccesarry_att(G, atts_to_keep = ['source', 'target', 'weight', 'geometry']):
    atts_name = list(set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True))))
    atts_to_remove = [i for i in atts_name if i not in ['source', 'target', 'weight', 'geometry']]
    for n1, n2, d in G.edges(data=True):
        for att in atts_to_remove:
            d.pop(att, None)
    return(G)

def interpolate_graph(G, max_length = 500, show_print=True):
    edges2length =nx.get_edge_attributes(G, 'length')
    long_edges = [key for key, value in edges2length.items() if value >= max_length]
    start_node_id = max(G.nodes) + 1
    for edge in long_edges:
        atts_edges = G.edges[edge]
        atts_nodes = {key: value for (key, value) in G.nodes[edge[0]].items()}
        nb_new_nodes = int(atts_edges['length'] // max_length)
        line = atts_edges['geometry']
        new_nodes = MultiPoint([Point(line.coords[0])] + [line.interpolate(i/(nb_new_nodes + 1), normalized=True) for i in range(1,nb_new_nodes+1)] + [Point(line.coords[-1])])
        int_nodes = new_nodes.geoms[1:-1].geoms
        for j in range(len(int_nodes)):
            atts_nodes['x'] = int_nodes[j].x
            atts_nodes['y'] = int_nodes[j].y
            atts_nodes['edge'] = edge
            G.add_node(start_node_id + j, **atts_nodes)

        new_edges = split(snap(line, new_nodes, 1), new_nodes).geoms
        parent_edge_length, parent_edge_weight = atts_edges['length'], atts_edges['weight']
        
        for i in range(len(new_edges)):
            atts_edges['geometry'] = new_edges[i]
            atts_edges['length'] = linestring_length(atts_edges['geometry'])
            atts_edges['weight'] = parent_edge_weight * atts_edges['length'] / parent_edge_length
            atts_edges['parent_edges'] = edge
            if i == 0:
                from_node = edge[0]
                to_node = start_node_id
            elif i == len(new_edges) - 1:
                from_node = start_node_id
                to_node = edge[1]
            else:
                from_node = start_node_id
                start_node_id += 1
                to_node = start_node_id
            G.add_edge(from_node, to_node, **atts_edges)
        start_node_id += 1
        G.remove_edge(edge[0], edge[1])
        
    if show_print:
        print("interpolate graph - interpolation distance: {} m - done".format(max_length))
    return(G)

def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
    return earth_radius * 2 * np.arcsin(np.sqrt(a))

def compute_edge_length(G, show_print=True):
    
    edges2geom = nx.get_edge_attributes(G, 'geometry')
    geom_edges = pd.DataFrame(edges2geom.items(),columns=['edge', 'geometry'])
    geom_edges['points'] = geom_edges.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
    geom_points = geom_edges.explode('points')
    geom_points[['lon', 'lat']] = pd.DataFrame(geom_points['points'].tolist(), index=geom_points.index)
    geom_points['dist'] = haversine(geom_points["lat"], geom_points["lon"], geom_points["lat"].shift(-1), geom_points["lon"].shift(-1))
    geom_points['dist'] = 1000 * geom_points['dist'] * (geom_points['edge'] == geom_points['edge'].shift(-1))
    edges2length = geom_points.groupby('edge')['dist'].sum().to_dict()
    nx.set_edge_attributes(G, edges2length, 'length')
    
    if show_print:
        print("Compute edge legnth done")
    return(G)

def fill_missing_geometry(G, show_print=True):
    for u, v in G.edges:
        atts = G.edges[u,v].keys()
        if 'geometry' not in atts:
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            edge_geometry = LineString((Point((u_node['x'], u_node['y'])), Point((v_node['x'], v_node['y']))))
            G.edges[u,v]['geometry'] = edge_geometry
            
    if show_print:
        print("Fill missing geometry done")
    return(G)

def process_graph(G, max_length = 500, save=True, folder_name="mm_input", show_print=True):
    if show_print:
        print_step("Start process graph")
        
    if save and (os.path.isfile(folder_name + '/graph.pickle')):
        G = load_param(folder_name, "graph", show_print=show_print)
        edges = nx.to_pandas_edgelist(G)
        edges['edge'] = list(zip(edges["source"], edges["target"]))
        set_id_to_edge = edges.set_index("edge_id")['edge'].to_dict()
    
    else:
        G = remove_unneccesarry_att(G)
        G = fill_missing_geometry(G, show_print=show_print)
        G = compute_edge_length(G, show_print=show_print)
        G = interpolate_graph(G, max_length = max_length, show_print=show_print)
        
        edges = nx.to_pandas_edgelist(G)
        edges['edge'] = list(zip(edges["source"], edges["target"]))
        edges['edge_id'] = range(len(edges))
        edges2id = edges.set_index("edge")['edge_id'].to_dict()
        set_id_to_edge = edges.set_index("edge_id")['edge'].to_dict()
        nx.set_edge_attributes(G, edges2id, 'edge_id')
        G = get_geohashes_of_edge(G, show_print=show_print)

        if save:
            save_param(folder_name, "graph", G)
    
    return(G, set_id_to_edge)

def print_step(msg):
    print("-"*len(msg))
    print(msg)
    print("-"*len(msg))
    return(None)

def format_result(gps, gps_map_matched):
    gps.reset_index(drop=True, inplace=True)
    gps_map_matched['shortest_path_nodes'] = gps_map_matched.apply(lambda row: row['map_match'][1], axis=1)
    gps_map_matched['edges'] = gps_map_matched.apply(lambda row: row['map_match'][0], axis=1)
    gps = gps[['lon', 'lat', 'trip_id']]
    gps['edge'] = gps_map_matched['edges'].explode().reset_index(drop=True)
    gps_map_matched = gps_map_matched["shortest_path_nodes"].reset_index()
    return(gps, gps_map_matched)

def node2edge_graph(graph_nx, show_print=True):
    edges = list(graph_nx.edges())
    edges2weight =nx.get_edge_attributes(graph_nx, 'weight')
    edges2id =nx.get_edge_attributes(graph_nx, 'edge_id')
    new_edges = []
    for from_node, to_node in edges:
        id_edge = edges2id[(from_node, to_node)]
        for i in graph_nx.in_edges(from_node):
            id_row = edges2id[i]
            weight = (edges2weight[(from_node, to_node)] + edges2weight[i]) / 2
            new_edges.append((id_row, id_edge, weight))
    new_graph = nx.DiGraph()
    new_graph.add_weighted_edges_from(new_edges)
    
    if show_print:
        print("Node to edge graph conversion done")
    return(new_graph)

def edge_graph_transistions(new_graph, max_weight, show_print=True):
    all_transition = []
    for id_node, ori_node in enumerate(list(new_graph.nodes)):
        one_edge_trans = nx.single_source_dijkstra_path_length(new_graph, ori_node, cutoff=max_weight)
        all_transition.append((ori_node, list(one_edge_trans.keys()), list(one_edge_trans.values())))
    all_transition = pd.DataFrame(all_transition, columns=['ori_edge', 'dest_edge', 'weight'])
    
    if show_print:
        print("Edge to edge transition computation done")
    return(all_transition)

def chunk_distance_computation(chunk_gps, candidate_edges, radius=150, alpha=0.1):
    dist_df = chunk_gps[["id", "geometry"]].join(candidate_edges, how='inner')
    dist_df.reset_index(drop=True, inplace=True)
    dist_df['dist'] = dist_df['geometry'].distance(dist_df['geometry_edge'])
    dist_df.drop(columns=['geometry', 'geometry_edge'], inplace=True)
    dist_df = dist_df[dist_df['dist'] < radius]
    dist_df['dist'] =  ((np.exp(-0.5* ((dist_df['dist'].values / 1000)/alpha)**2)/(np.sqrt(2*np.pi) * alpha))*10).astype(np.int8)
    return(dist_df[['id','edge','dist']])

def emission_matrix(gps, G, dic_geohash, edge_geohash_dict, alpha = 0.1, radius = 150, chunk=True, nb_chunks = 20, show_print=True):  
    if show_print:
        print_step("Start process emission")
    geom_df = (nx.to_pandas_edgelist(G)).rename(columns={'edge_id': 'edge'})
    geom_df = gpd.GeoDataFrame(geom_df[['edge', 'geometry']], geometry=geom_df['geometry'], crs=4326)
    geom_df.to_crs(3035, inplace=True)
    candidate_edges = pd.DataFrame.from_dict(edge_geohash_dict.items())
    candidate_edges.columns=['geohash', 'edge']
    candidate_edges['geohash_int'] = candidate_edges['geohash'].map(dic_geohash)
    candidate_edges.drop(columns=["geohash"], inplace=True)
    candidate_edges = candidate_edges.explode("edge")
    candidate_edges.reset_index(drop=True, inplace=True)
    candidate_edges = candidate_edges.astype(np.uint32)
    candidate_edges = candidate_edges.merge(geom_df, on='edge')
    candidate_edges.set_index('geohash_int', inplace=True)
    candidate_edges.rename(columns={'geometry': 'geometry_edge'}, inplace=True)
    
    if chunk:
        gps_splited = np.array_split(gps, nb_chunks)
        dist_df = []
        for idx, chunk_gps in enumerate(gps_splited):
            dist_df.append(chunk_distance_computation(chunk_gps, candidate_edges, radius=radius, alpha=alpha))
        dist_df = pd.concat(dist_df)

    else:
        dist_df = gps[["id", "geometry"]].join(candidate_edges, how='inner')
        dist_df.reset_index(drop=True, inplace=True)
        dist_df['dist'] = dist_df['geometry'].distance(dist_df['geometry_edge'])
        dist_df.drop(columns=['geometry', 'geometry_edge'], inplace=True)
        dist_df = dist_df[dist_df['dist'] < radius]
        dist_df['dist'] =  ((np.exp(-0.5* ((dist_df['dist'].values / 1000)/alpha)**2)/(np.sqrt(2*np.pi) * alpha))*10).astype(np.int8)
        
    nb_rows, nb_cols = int(dist_df['id'].max() + 1), len(G.edges())
    emission_matrix = csr_matrix((dist_df['dist'].values, ( dist_df['id'].values, dist_df['edge'].values)), shape=(nb_rows,nb_cols))
    print('done') 
    return(emission_matrix)

def transition_matrix(graph_nx, max_weight, beta = 1/500, save=True, folder_name="mm_input", show_print=True):
    if show_print:
        print_step("Start process transition")
        
    if save and (os.path.isfile(folder_name + '/transition_matrix.pickle')):
        transition_matrix = load_param(folder_name, "transition_matrix", show_print=show_print)
        return(transition_matrix)
    else:
        new_graph = node2edge_graph(graph_nx, show_print=show_print)
        all_transition = edge_graph_transistions(new_graph, max_weight, show_print=show_print)
        all_transition = all_transition.explode(['dest_edge', 'weight']).reset_index(drop=True).astype(np.int64)
        int_max_edge = max(all_transition[["ori_edge", "dest_edge"]].max()) + 1
        row = all_transition['ori_edge'].values
        col = all_transition['dest_edge'].values
        data = np.exp(- beta * all_transition['weight'].values)
        transition_matrix = csr_matrix((data, (row, col)), shape=(max(row)+1, max(col)+1))
        transition_matrix = (transition_matrix * 100).astype(np.int8)
        if save:
            save_param(folder_name, "transition_matrix", transition_matrix)
        return(transition_matrix)
    
def get_all_neigbors(hash_central, dir_geo, dic_neighbor, nb_steps = 19):
    all_keys = dic_neighbor.keys()
    all_neigbors_dir = [hash_central]
    i = 0
    for i in range(nb_steps):
        if (hash_central, dir_geo) in all_keys:
            neigbhor_dir = dic_neighbor[(hash_central, dir_geo)]
        else:
            neigbhor_dir = pgh.get_adjacent(hash_central, dir_geo)
            dic_neighbor[(hash_central, dir_geo)] = neigbhor_dir
            
            all_neigbors_dir.append(neigbhor_dir)
        hash_central = neigbhor_dir
    return(all_neigbors_dir, dic_neighbor)

def compute_neighbors(hash_central, nb_steps_lon, nb_steps_lat, dic_neighbor):
    all_neigbors_dir_north, dic_neighbor = get_all_neigbors(hash_central, "top",dic_neighbor, nb_steps=nb_steps_lat)
    all_neigbors_dir_south, dic_neighbor = get_all_neigbors(hash_central, "bottom", dic_neighbor, nb_steps=nb_steps_lat)
    vertical_geoshes = all_neigbors_dir_south + all_neigbors_dir_north
    all_geohashes = vertical_geoshes.copy()
    for hash_v in vertical_geoshes:
        all_neigbors_dir_west, dic_neighbor = get_all_neigbors(hash_v, "left", dic_neighbor, nb_steps=nb_steps_lon)
        all_neigbors_dir_est, dic_neighbor  = get_all_neigbors(hash_v, "right", dic_neighbor, nb_steps=nb_steps_lon)
        all_geohashes.extend(all_neigbors_dir_west + all_neigbors_dir_est)
    
    return(all_geohashes, dic_neighbor)

def get_all_emission_tiles(geohashes, edge, nb_steps_lon, nb_steps_lat, dic_neighbor, dic_cand_edges): 
    all_geohashes_tiles = []
    dic_neighbor = {}
    for i in geohashes:
        sub_geohashes, dic_neighbor = compute_neighbors(i, nb_steps_lon, nb_steps_lat ,dic_neighbor)
        all_geohashes_tiles.extend(sub_geohashes)
    all_geohashes_tiles = list(set(all_geohashes_tiles))
    
    keys_geohash = dic_cand_edges.keys()
    for i in all_geohashes_tiles:
        if i not in keys_geohash:
            dic_cand_edges[i] = [edge]
        else:
            dic_cand_edges[i] += [edge]
    return(all_geohashes_tiles)

def get_number_tiles(radius):
    width, height = 38.2, 19.1
    nb_lon_steps = int((radius // width) + 1)
    nb_lat_steps = int((radius // height) + 1)
    return(nb_lat_steps, nb_lon_steps)

def get_candidate_edges_dictionary(G, radius = 150, show_print=True):
    edges2id =nx.get_edge_attributes(G, 'edge_id')
    edge2geohash = nx.get_edge_attributes(G, 'geohashes')
    edge_geohash = pd.DataFrame(edge2geohash.items(),columns=['edge', 'geohash'])
    edge_geohash['edge_id'] = edge_geohash['edge'].map(edges2id)
    dic_cand_edges = {}
    dic_neighbor = {}
    nb_lat_steps, nb_lon_steps = get_number_tiles(radius)
    edge_geohash['geohash_bis'] = edge_geohash.apply(lambda row : get_all_emission_tiles(row['geohash'], row['edge_id'], nb_lon_steps, nb_lat_steps, dic_neighbor, dic_cand_edges), axis=1)
    if show_print:
        print("Get candidate edges for all geohashes done")
    return(dic_cand_edges)

def load_param(folder_name, param_name, show_print=True):
    with open('{}/{}.pickle'.format(folder_name, param_name), 'rb') as handle:
        param = pickle.load(handle)
    if show_print:
        print("Load existing {} done".format(param_name))
    return(param)

def save_param(folder_name, param_name, param):
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    with open('{}/{}.pickle'.format(folder_name, param_name), 'wb') as handle:
        pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return(None)

def process_candidate_edges_dictionary(G, radius = 150, save=True, folder_name="mm_input", show_print=True):
    if save and (os.path.isfile(folder_name + '/cand_edges.pickle')):
        dic_cand_edges = load_param(folder_name, "cand_edges", show_print=show_print)

    else:
        dic_cand_edges = get_candidate_edges_dictionary(G, radius = radius, show_print=show_print)

        if save and (~os.path.isfile(folder_name + '/cand_edges.pickle')):
            save_param(folder_name, "cand_edges", dic_cand_edges)
    return(dic_cand_edges)

def process_gps(gps, cand_edge):
    gps.reset_index(drop=True, inplace=True)
    gps['id'] = range(len(gps))
    gps["id"] = gps["id"].astype(np.uint32)
    gps['geohash']   = pygeohash_fast.encode_many(gps['lon'].values, gps['lat'].values, 8)
    all_geohashes = set(gps['geohash'] )
    dic_geohash = dict(zip(all_geohashes, range(len(all_geohashes))))
    edge_geohash_dict = {k: cand_edge[k] for k in cand_edge.keys() & all_geohashes}

    gps['edge'] = gps['geohash'].map(edge_geohash_dict)
    gps['geohash_int'] = gps['geohash'].map(dic_geohash)
    gps.set_index('geohash_int', inplace=True)
    gps = gpd.GeoDataFrame(gps, geometry=gpd.points_from_xy(gps['lon'], gps['lat']), crs=4326)
    gps.to_crs(3035, inplace=True)
    gps['points'] = list(zip(gps['lon'], gps['lat']))
    
    gps_map_matched = gps.groupby('trip_id').agg({'points' : lambda row: np.array(list(row), dtype=float),  
                                            "id":['first', 'last'], 
                                            "edge":  sum})
    gps_map_matched.columns = ['traj', 'first_emission', 'last_emission', 'sub_edges']
    gps_map_matched = gps_map_matched[gps_map_matched['sub_edges']!=0]
    gps_map_matched['sub_edges'] = gps_map_matched['sub_edges'].apply(lambda row: list(set(row)))
    gps.drop(columns=['geohash', 'points', "edge"], inplace=True)
    return(gps, gps_map_matched, dic_geohash, edge_geohash_dict)

"""
Map matching functions
"""

@numba.jit(nopython=True)
def fast_viterbi(obs, states, start_p, trans_p, emit_p):
    nb_obs = obs.shape[0]
    nb_states = states.shape[0]

    V = np.zeros((nb_obs, nb_states))
    path = np.zeros((1,nb_states))

    for idx, y in enumerate(states):
        V[0, y] = start_p[y] * emit_p[ obs[0],y]
        path[0, idx] = y

    for itime in range(1, nb_obs):   
        states_cand = np.where(emit_p[ obs[itime],:] != 0)[0]
        newpath = np.zeros((itime + 1 ,states_cand.shape[0]))
        states_bis =np.where(V[itime-1, :] != 0)[0]
        for idx, y in enumerate(states_cand):
            prob = 0
            for y0 in states_bis:
                if V[itime-1, y0] * trans_p[y0,y] * emit_p[ obs[itime],y] > prob:
                    prob = V[itime-1, y0] * trans_p[y0,y] * emit_p[ obs[itime],y]
                    state = y0
            V[itime, y] = prob
            if prob>0:
                row = np.where(path[-1,:] == state)[0][0]
                newpath[:itime, idx] = path[:, row]
                newpath[-1, idx] = y
        path = newpath

    idx_max = np.argmax(V[itime,:])
    (prob, state) = V[itime, idx_max], idx_max
    if prob !=0:
        row = np.where(path[-1,:] == state)[0][0]
        seq_nodes = path[:, row]
    else:
        path = np.zeros((1,nb_states))
        seq_nodes = np.zeros((1,nb_states))[:,0]
    return (V,prob, seq_nodes,path)

def one_trajectory_mapmatching(GPS_traj, graph, transition_matrix, emission_matrix, set_id_to_edge, sub_edges, start, end):
    try:
        emit_p = (emission_matrix[start:end+1, : ][:, sub_edges].toarray())/10
        trans_p = (transition_matrix[sub_edges, :][:, sub_edges].toarray())/100
        start_p = np.ones(len(sub_edges))/len(sub_edges)
        obs = np.array([i for i in range(len(GPS_traj))])
        states = np.array([i for i in range(len(sub_edges))])
        (V,prob, state,path) = fast_viterbi(obs, states, start_p, trans_p, emit_p)
        if prob != 0:
            edges = [set_id_to_edge[sub_edges[int(i)]] for i in state]
            edges_without_redundancy = [i[0] for i in itertools.groupby(edges)]
            path = [nx.shortest_path(graph, edges_without_redundancy[i][1], edges_without_redundancy[i+1][0]) for i in range(len(edges_without_redundancy) - 1)]
            path = [edges_without_redundancy[0][0]] + [item for sublist in path for item in sublist] + [edges_without_redundancy[-1][1]]
            return(edges, path)
        else:
            return([np.nan]*len(GPS_traj),[])
    except:
        return(["Problem"], ["Problem"])

def map_matching_precomputation(G, beta=1/500, radius=150, save=True, folder_name="mm_input", show_print=True):
    G_mapmatching, set_id_to_edge = process_graph(G, max_length=500, save=save, folder_name=folder_name, show_print=show_print)
    trans = transition_matrix(G_mapmatching, 120, beta=beta, save=save, folder_name=folder_name, show_print=show_print)
    edge_geohash_dict = process_candidate_edges_dictionary(G_mapmatching, radius = radius, save=True, folder_name=folder_name, show_print=show_print)
    return(G_mapmatching, trans, edge_geohash_dict, set_id_to_edge)

def map_matching_gps(gps, G_mapmatching, trans, edge_geohash_dict, set_id_to_edge, alpha=0.1, radius=150):
    gps, gps_map_matched, dic_geohash, edge_geohash_dict = process_gps(gps, edge_geohash_dict)
    emit = emission_matrix(gps, G_mapmatching, dic_geohash, edge_geohash_dict, alpha = alpha, radius = radius)
    gps_map_matched['map_match'] = gps_map_matched.apply(lambda row: one_trajectory_mapmatching(row['traj'], G_mapmatching, trans, emit, set_id_to_edge, row['sub_edges'], row['first_emission'], row['last_emission']), axis=1)
    gps, gps_map_matched = format_result(gps, gps_map_matched)
    return(G_mapmatching, gps, gps_map_matched)

def optimized_map_matching(G, gps, save=True, radius = 150, alpha = 0.1, beta=1/500, folder_name="mm_input", show_print=True):
    G_mapmatching, trans, edge_geohash_dict, set_id_to_edge = map_matching_precomputation(G, beta=beta, radius=radius, save=save, folder_name=folder_name, show_print=show_print)
    G_mapmatching, gps, gps_map_matched = map_matching_gps(gps, G_mapmatching, trans, edge_geohash_dict, set_id_to_edge, alpha=alpha, radius=radius)
    return(G_mapmatching, gps, gps_map_matched)

G = ox.graph_from_place('Hyderabad', network_type = 'drive', simplify=False)
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
G = ox.simplify_graph(G)
G = nx.DiGraph(G)
edges2tt =nx.get_edge_attributes(G, 'travel_time')
nx.set_edge_attributes(G, edges2tt, 'weight')

gps = pd.DataFrame([(1, 78.35139, 17.44619),
                    (1, 78.35050, 17.44705),
                    (1, 78.34880, 17.44868),
                    (1, 78.34841, 17.44944),
                    (1, 78.34898, 17.44890),]
                   , columns = ['trip_id', 'lon', 'lat'])

G_mapmatching, gps_nearest, gps_map_matched = optimized_map_matching(G, gps)
print(gps_nearest)
print(gps_map_matched)