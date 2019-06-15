from scipy.spatial import Delaunay
from igraph import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

def get_actions(actions_array, position_array):
    #rotulos de acoes
    actions_labels = ['Domínio','Passe','Drible','Finalização-chute','Finalização-cabeca','Desarme (inf)','Desarme (sup)','Defesa Goleiro','Saida do Goleiro','Tiro-de-meta','Lateral','Escanteio','Impedimento','Falta','Gol', 'Condução']

    dummies = pd.get_dummies(actions_array['actions'])
    dummies = dummies.T.reindex(np.arange(0,16)).T.fillna(0)
    dummies.columns = actions_labels

    actions_array = pd.concat([actions_array, pd.DataFrame(['team1' if i <= 14 else 'team2' for i in actions_array['player']], columns=['team'])], axis=1)

    sorted_x = np.sort(np.unique(position_array[:,np.arange(1,len(position_array[0]),2)]))
    sorted_y = np.sort(np.unique(position_array[:,np.arange(2,len(position_array[0]),2)]))

    min_x = sorted_x[1]
    min_y = sorted_y[1]
    max_x = sorted_x[-1]
    max_y = sorted_y[-1]

    center_x = (min_x + max_x)/2

    third_min, third_max = np.take(np.linspace(min_x, max_x,4), [1,2])

    return (center_x, third_min, actions_array)

def constroi_grafo_delaunay(team1: tuple, team2: tuple, distance):

    """
        Gera o grafo do time1 (azul) e do time2 (vermelho), alem de retirar arestas que passam perto de um
        jogador do time adversario menores que a distancia fornecida.

        Parameters
        ----------
            team1
                triangulacao de delaunay do time1
            team2
                triangulacao de delaunay do time2
            distance
                distancia utilizada para verificar o quao longe uma aresta esta do time adversario
        Return
        ------
            tuple
                tupla de coordenadas do time1 e time2 que sao maiores que a distancia 
            g1
                grafo do time1
            g2 
                grafo do time 2
    """

    g1 = Graph()
    g1.add_vertices(11)

    g2 = Graph()
    g2.add_vertices(11)

    vertex_graphs = team1[1].simplices
    coord = [team1[1].points[x] for x in team1[1].simplices]

    final_points_t1_1 = []
    final_points_t1_2 = []

    for i in range(len(coord)):
        coords = coord[i]
        vertex = vertex_graphs[i]

        v0 = vertex[0]
        v1 = vertex[1]
        v2 = vertex[2]

        p0 = np.array(coords[0])
        p1 = np.array(coords[1])
        p2 = np.array(coords[2])

        total_d0 = []
        total_d1 = []
        total_d2 = []

        for points in team2[0]:
            d0 = abs((np.cross(p1-p0, p1-points)/np.linalg.norm(p1-p0)))
            d1 = abs((np.cross(p2-p1, p2-points)/np.linalg.norm(p2-p1)))
            d2 = abs((np.cross(p0-p2, p0-points)/np.linalg.norm(p0-p2)))
            if d0 < distance:
                total_d0.append(d0)
            if d1 < distance:
                total_d1.append(d1)
            if d2 < distance:
                total_d2.append(d2)

        if not total_d0:
            final_points_t1_1.append(p0)
            final_points_t1_2.append(p1)
            g1.add_edge(v0, v1)
        if not total_d1:
            final_points_t1_1.append(p1)
            final_points_t1_2.append(p2)
            g1.add_edge(v1, v2)
        if not total_d2:
            final_points_t1_1.append(p2)
            final_points_t1_2.append(p0)
            g1.add_edge(v0, v2)

    vertex_graphs = team2[1].simplices
    coord = [team2[1].points[x] for x in team2[1].simplices]

    final_points_t2_1 = []
    final_points_t2_2 = []

    for i in range(len(coord)):
        coords = coord[i]
        vertex = vertex_graphs[i]

        v0 = vertex[0]
        v1 = vertex[1]
        v2 = vertex[2]

        p0 = np.array(coords[0])
        p1 = np.array(coords[1])
        p2 = np.array(coords[2])

        total_d0 = []
        total_d1 = []
        total_d2 = []

        for points in team1[0]:
            d0 = abs((np.cross(p1-p0, p1-points)/np.linalg.norm(p1-p0)))
            d1 = abs((np.cross(p2-p1, p2-points)/np.linalg.norm(p2-p1)))
            d2 = abs((np.cross(p0-p2, p0-points)/np.linalg.norm(p0-p2)))
            if d0 < distance:
                total_d0.append(d0)
            if d1 < distance:
                total_d1.append(d1)
            if d2 < distance:
                total_d2.append(d2)

        if not total_d0:
            final_points_t2_1.append(p0)
            final_points_t2_2.append(p1)
            g2.add_edge(v0, v1)
        if not total_d1:
            final_points_t2_1.append(p1)
            final_points_t2_2.append(p2)
            g2.add_edge(v1, v2)
        if not total_d2:
            final_points_t2_1.append(p2)
            final_points_t2_2.append(p0)
            g2.add_edge(v0, v2)

    return (final_points_t1_1, final_points_t1_2, final_points_t2_1, final_points_t2_2), g1, g2

def filtrar_dados(positions_array):
    """ Filtra dados do arquivo .2d para vetores numpy
        Obtem posicao de 24 jogadores em um dado frame

    Parameters
    ----------
        positions_array:
            todos os dados carregados do arquivo
    Return
    ------
        tuple
            tupla de coordenadas do time1 e time2
    """
    array_shape = positions_array.shape

    num_players = int((array_shape[1]-1)/4) - 1
    num_frames = array_shape[0]

    players_pos_team1 = np.zeros(shape=(array_shape[0], num_players, 2))
    players_pos_team2 = np.zeros(shape=(array_shape[0], num_players, 2))

    for i in range(num_players):
        players_pos_team1[:,i] = positions_array[:, (2*i)+1:(2*i)+3]
        players_pos_team2[:,i] = positions_array[:, (2*num_players-1)+(2*i):(2*num_players-1)+(2*i)+2]

    return (players_pos_team1, players_pos_team2)

def data_manipulation(teams: tuple, actions: tuple):
    """

        Retorna frames que ocorreram alguma acao

    """

    players_pos_team1 = teams[0]
    players_pos_team2 = teams[1]

    center_x = actions[0]
    third_min = actions[1]
    actions_array = actions[2]

    if np.mean(np.sort(np.unique(players_pos_team1[0,:,0]))[1:]) > np.mean(np.sort(np.unique(players_pos_team2[0,:,0]))[1:]):
        start_side_team1 = 'right'
        start_side_team2 = 'left'
    else:
        start_side_team1 = 'left'
        start_side_team2 = 'right'

    activity = []
    region = []
    for team, x in zip(actions_array['team'], actions_array['x']):
        if team == 'team1':
            start_side = start_side_team1
        else:
            start_side = start_side_team2

        if start_side == 'right' and x < actions[0]:
    #         activity.append('attack')
            if x < actions[1]:
                region.append('danger')
            else:
                region.append('neutral')
        else:
    #         activity.append('defense')
            region.append('defense field')

    actions_array = pd.concat([actions_array, pd.DataFrame(region, columns=['region'])], axis=1)

    return actions_array

def get_frame_position(teams: tuple, frame: int):

    players_pos_team1 = teams[0]
    players_pos_team2 = teams[1]

    in_game_team1 = players_pos_team1[frame][np.all(players_pos_team1[frame] >= 0, axis=1)]
    in_game_team2 = players_pos_team2[frame][np.all(players_pos_team2[frame] >= 0, axis=1)]

    return (in_game_team1, in_game_team2)

def plot_all_players_delaunay(team1: tuple, team2: tuple):
    """ Plota todos os jogadores em dois grafos, representando seus respectivos times,
         montados por triangulacao de Delaunay

    Parameters
    ----------
        team1: tuple
            dupla de dados do 1 time contendo os dados filtrados e triangularizados
        team2: tuple
            dupla de dados do 2 time contendo os dados filtrados e triangularizados
    """
    # especifica pontos para plotagem dos triangulos
    plt.triplot(team1[0][:, 0], team1[0][:, 1], team1[1].simplices.copy(), c= 'b')
    plt.triplot(team2[0][:, 0], team2[0][:, 1], team2[1].simplices.copy(), c= 'r')
    # seta os limites de x e y
    plt.xlim(0,120)
    plt.ylim(0,80)
    # especifica grafico de dispersão
    plt.scatter(team1[0][:, 0], team1[0][:, 1], c='b')
    plt.scatter(team2[0][:, 0], team2[0][:, 1], c='r')
    plt.show()

def plot_final_points(final_points: tuple):
    for i in range(len(final_points[1])):
        x1, y1 = final_points[0][i]
        x2, y2 = final_points[1][i]
        saida1 = [x1, x2]
        saida2 = [y1, y2]
        plt.plot(saida1, saida2, marker = 'o', c='b')

    for i in range(len(final_points[2])):
        x1, y1 = final_points[2][i]
        x2, y2 = final_points[3][i]
        saida1 = [x1, x2]
        saida2 = [y1, y2]
        plt.plot(saida1, saida2, marker = 'o', c='r')
    plt.xlim(0,120)
    plt.ylim(0,80)
    plt.show()

def gera_dados(g1, g2):

    """
        Recebe os dois grafos dos dois times e retorna as caracteristicas dos grafos
    """

    g1.simplify()

    #graus do g1
    graus_t1 = g1.degree()

    #excentricidade de g1
    ecc_t1 = g1.eccentricity()

    #centralidade de g1
    cent_t1 = g1.evcent()

    g2.simplify()

    #graus do g1
    graus_t2 = g2.degree()

    #excentricidade de g1
    ecc_t2 = g2.eccentricity()

    #centralidade de g1
    cent_t2 = g2.evcent()

    return (graus_t1, ecc_t1, cent_t1), (graus_t2, ecc_t2, cent_t2)

def action_filter(table, team=None, region=None, action=None, distance=0.5):
    if team == None:
        team = table['team']

    if region == None:
        region = table['region']
    
    if action == None:
        action = table['actions']

    selection_table = table[np.logical_and.reduce((table['team'] == team, table['region'] == region, table['actions'] == action))]

    ecc_team1 = []
    ecc_team2 = []

    cent_team1 = []
    cent_team2 = []

    graus_team1 = []
    graus_team2 = []

    for frame in selection_table.frame:
        teams = get_frame_position(teams_data, frame)
        data_graph1 = (teams[0], Delaunay(teams[0]))
        data_graph2 = (teams[1], Delaunay(teams[1]))
        final_points, g1, g2 = constroi_grafo_delaunay(data_graph1, data_graph2, distance)
        (graus_t1, ecc_t1, cent_t1), (graus_t2, ecc_t2, cent_t2) = gera_dados(g1, g2)
        ecc_team1.append(ecc_t1)
        ecc_team2.append(ecc_t2)
        cent_team1.append(cent_t1)
        cent_team2.append(cent_t2)
        graus_team1.append(graus_t1)
        graus_team2.append(graus_t2)
    
    return ecc_team1, ecc_team2, cent_team1, cent_team2, graus_team1, graus_team2

if __name__ == "__main__":
    # carrega dados do arquivo
    positions_array = np.loadtxt('data/REDMACT1suav.2d').astype('int')

    #carrega acoes
    labels = ['frame', 'player', 'x', 'y', 'actions', 'status']
    actions_array = pd.DataFrame(np.loadtxt('data/REDMACT1.ant').astype('int'), columns=labels)

    #pega as acoes do jogo
    actions = get_actions(actions_array, positions_array)

    teams_data = filtrar_dados(positions_array)

    table = data_manipulation(teams_data, actions)

    actions_labels = ['Domínio','Passe','Drible','Finalização-chute','Finalização-cabeca','Desarme (inf)','Desarme (sup)','Defesa Goleiro','Saida do Goleiro','Tiro-de-meta','Lateral','Escanteio','Impedimento','Falta','Gol', 'Condução']
    mapping = {key: value for (key, value) in enumerate(actions_labels)}
    table = table.replace({'actions': mapping})

    ecc_team1, ecc_team2, cent_team1, cent_team2, graus_team1, graus_team2 = action_filter(table, team='team1', region='danger', distance=0.8)

    print(np.mean(graus_team1))
    print(np.mean(graus_team2))