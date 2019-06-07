from scipy.spatial import Delaunay
from igraph import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def constroi_grafo_delaunay(team1: tuple, team2: tuple, distance):

	g = Graph()
	g.add_vertices(11)

	vertex_graphs = team2[1].simplices
	coord = [team2[1].points[x] for x in team2[1].simplices]

	final_points1 = []
	final_points2 = []

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
			final_points1.append(p0)
			final_points2.append(p1)
			g.add_edge(v0, v1)
		if not total_d1:
			final_points1.append(p1)
			final_points2.append(p2)
			g.add_edge(v1, v2)
		if not total_d2:
			final_points1.append(p2)
			final_points2.append(p0)
			g.add_edge(v0, v2)

	return (final_points1, final_points2), g

def filtrar_dados(data_array, frame: int):
	""" Filtra dados do arquivo .2d para vetores numpy
		Obtem posicao de 24 jogadores em um dado frame
	
	Parameters
	----------
		data_array:
			todos os dados carregados do arquivo
		frame: int
			o frame em questao para obtecao dos dadodos
	Return
	------
		tuple
			tupla de coordenadas do time1, time2 e a quantidade todal de frames da partida
	"""
	array_shape = data_array.shape

	num_players = int((array_shape[1]-1)/4) - 1
	num_frames = array_shape[0]

	players_pos_team1 = np.zeros(shape=(array_shape[0], num_players, 2))
	players_pos_team2 = np.zeros(shape=(array_shape[0], num_players, 2))

	for i in range(num_players):
		players_pos_team1[:,i] = data_array[:, (2*i)+1:(2*i)+3]
		players_pos_team2[:,i] = data_array[:, (2*num_players-1)+(2*i):(2*num_players-1)+(2*i)+2]

	in_game_team1 = players_pos_team1[frame][np.all(players_pos_team1[frame] >= 0, axis=1)]
	in_game_team2 = players_pos_team2[frame][np.all(players_pos_team2[frame] >= 0, axis=1)]
	
	return (in_game_team1, in_game_team2, num_frames)

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
	plt.ylim(0,60)
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
		plt.plot(saida1, saida2, marker = 'o', c='r')
	plt.show()

def gera_dados(grafo):
	grafo.simplify()

	#graus do grafo
	graus = grafo.degree()

	#excentricidade
	ecc = grafo.eccentricity()

	#centralidade
	cent = grafo.evcent()

	return (graus, ecc, cent)

if __name__ == "__main__":
	# carrega dados do arquivo
	data_array = np.loadtxt('data/CapSpoT2Suav.2d').astype('int')

	# gera vetores numpy para o frame especificado
	teams = filtrar_dados(data_array, 3000)

	# cria tuplas de vetores numpy do dado frame e seu grafo delaunay
	data_graph1 = (teams[0], Delaunay(teams[0]))
	data_graph2 = (teams[1], Delaunay(teams[1]))

	# constroi o grafo por triangulação de Delaunay e retorna os "final_points" de cada time
	final_points, g = constroi_grafo_delaunay(data_graph1, data_graph2, 1.5)

	# plota todos os jogares em dois grafos de cada time
	plot_all_players_delaunay(data_graph1, data_graph2)

	# plota os final_points
	plot_final_points(final_points)

	# gera os dados do grafo
	dados = gera_dados(g)