from scipy.spatial import Delaunay
from igraph import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics



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
    
    #pagerank de g2
    pagerank_t1 = g1.pagerank()
    
    evcent_t1 = g1.evcent()
    

    g2.simplify()

    #graus do g2
    graus_t2 = g2.degree()

    #excentricidade de g2
    ecc_t2 = g2.eccentricity()

    #centralidade de g2
    cent_t2 = g2.evcent()
    
    #pagerank de g2
    pagerank_t2 = g2.pagerank()
    
    evcent_t2 = g2.evcent()

    return (graus_t1, ecc_t1, cent_t1, pagerank_t1, evcent_t1), (graus_t2, ecc_t2, cent_t2, pagerank_t2, evcent_t2)


def filter_table(table, team=None, region=None, action=None, distance=0):
    if team == None:
        team = table['team']

    if region == None:
        region = table['region']
    
    if action == None:
        action = table['actions']

    selection_table = table[np.logical_and.reduce((table['team'] == team, table['region'] == region, table['actions'] == action))]
    
    return selection_table
    

def bulk_properties(table, teams_data, team=None, region=None, action=None, distance=0):

    selection_table = filter_table(table, team=team, region=region, action=action, distance=distance)
        
    graus_team1 = []
    graus_team2 = []
    
    ecc_team1 = []
    ecc_team2 = []

    cent_team1 = []
    cent_team2 = []

    pagerank_team1 = []
    pagerank_team2 = []
    
    evcent_team1 = []
    evcent_team2 = []
    
    layout_team1 = []
    layout_team2 = []

    for frame in selection_table.frame:
        teams = get_frame_position(teams_data, frame)
        data_graph1 = (teams[0], Delaunay(teams[0]))
        data_graph2 = (teams[1], Delaunay(teams[1]))
        try:
            final_points, g1, g2 = constroi_grafo_delaunay(data_graph1, data_graph2, distance)
        except:
            continue
        (graus_t1, ecc_t1, cent_t1, pagerank_t1, evcent_t1), (graus_t2, ecc_t2, cent_t2, pagerank_t2, evcent_t2) = gera_dados(g1, g2)
        graus_team1.append(graus_t1)
        graus_team2.append(graus_t2)
        ecc_team1.append(ecc_t1)
        ecc_team2.append(ecc_t2)
        cent_team1.append(cent_t1)
        cent_team2.append(cent_t2)
        pagerank_team1.append(pagerank_t1)
        pagerank_team2.append(pagerank_t2)
        evcent_team1.append(evcent_t1)
        evcent_team2.append(evcent_t2)
        
    return (graus_team1, ecc_team1, cent_team1, pagerank_team1, evcent_team1),(graus_team2, ecc_team2, cent_team2, pagerank_team2, evcent_team2)

    
#     return (graus_team1, ecc_team1, cent_team1),(graus_team2, ecc_team2, cent_team2)


def contatenate_properties(properties, axis=1):
    array_prop = np.array(properties[0])
    for prop in properties[1:]:
        try:
            array_prop = np.concatenate((array_prop, np.array(prop)), axis=axis)
        except:
            pass
    return array_prop


def get_prop_both_teams(table, teams_data, region=None, action=None, distance=0):
    team = 'team1'
    properties_team1 = np.array(bulk_properties(table, teams_data, team=team, region=region, action=action, distance=distance)[0])
    team = 'team2'
    properties_team2 = np.array(bulk_properties(table, teams_data, team=team, region=region, action=action, distance=distance)[1])
    try:
        return np.concatenate((properties_team1, properties_team2), axis=1)
    except:
        if len(properties_team1.shape) == 3:
            return properties_team1
        elif len(properties_team2.shape) == 3:
            return properties_team2
        else:
            return None
        
def get_prop_all_games(files_ant, files_2d, region=None, action=None, distance=0):
    properties = []
    for file_ant, file_2d in zip(files_ant, files_2d):
        # carrega dados do arquivo
        data_array = np.loadtxt(file_2d).astype('int')
        
        teams_data = filtrar_dados(data_array)

        #carrega acoes
        labels = ['frame', 'player', 'x', 'y', 'actions', 'status']
        actions_array = pd.DataFrame(np.loadtxt(file_ant).astype('int'), columns=labels)

        #pega as acoes do jogo
        actions = get_actions(actions_array, data_array)

        teams_data = filtrar_dados(data_array)

        table = data_manipulation(teams_data, actions)

        actions_labels = ['Domínio','Passe','Drible','Finalização-chute','Finalização-cabeca','Desarme (inf)','Desarme (sup)','Defesa Goleiro','Saida do Goleiro','Tiro-de-meta','Lateral','Escanteio','Impedimento','Falta','Gol', 'Condução']
        mapping = {key: value for (key, value) in enumerate(actions_labels)}
        table = table.replace({'actions': mapping})
        
        properties.append(get_prop_both_teams(table, teams_data, region=region, action=action, distance=distance))
    
    return contatenate_properties(properties)

def evaluate_list(files_ant, files_2d, region=[], action=[], distance=0):
    properties = []
    if len(region)>1:
        evaluated_list = region
        for value in evaluated_list:
            properties.append(contatenate_properties(get_prop_all_games(files_ant, files_2d, region=value, distance=distance)))
        return properties
    elif len(action)>1:
        evaluated_list = action
        for value in evaluated_list:
            properties.append(contatenate_properties(get_prop_all_games(files_ant, files_2d, action=value, distance=distance)))
        return properties
    else:
        print("Error: Need specify one list as parameter")
        
def evaluate_prop(clf, X, y, classes_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    cross_acc = cross_val_score(clf, X_train, y_train, cv=3)
    clf.fit(X_train, y_train)
    properties_idx = np.arange(0,X.shape[1]+1,11)
    properties_relevance = []
    for i in range(len(properties_idx)-1):
        properties_relevance.append(np.mean(clf.feature_importances_[properties_idx[i]:properties_idx[i+1]])*11)
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    matrix = plot_confusion_matrix(y_test, y_pred, classes=np.array(classes_names))
    return cross_acc, properties_relevance, acc, recall, matrix, recall
        
def balance_data(X, y):
    labels, counts = np.unique(y, return_counts=True)
    min_elements = min(counts)

    X_balanced = np.zeros(shape=(min_elements*len(labels), X.shape[1]))
    y_balanced = np.zeros(shape=(min_elements*len(labels)))

    for i,label in enumerate(labels):
        idx = y==label

        X_temp = X[idx]
        np.random.seed(0)
        np.random.shuffle(X_temp)

        X_balanced[i*min_elements:(i+1)*min_elements] = X_temp[:min_elements]
        y_balanced[i*min_elements:(i+1)*min_elements] = [i]*min_elements
        
    return X_balanced, y_balanced



def generate_labels(properties):
    labels = []
    for i in range(len(properties)):
        labels.append(np.full((len(properties[i])), i))
    return labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred).astype(int)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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

    ecc_team1, ecc_team2, cent_team1, cent_team2, graus_team1, graus_team2 = bulk_properties(table, team='team1', region='danger', distance=0.8)

    print(np.mean(graus_team1))
    print(np.mean(graus_team2))