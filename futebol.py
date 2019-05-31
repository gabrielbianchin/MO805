import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import Delaunay
from igraph import *

data_array = np.loadtxt('CapSpoT2Suav.2d').astype('int')

array_shape = data_array.shape

num_players = int((array_shape[1]-1)/4) - 1
num_frames = array_shape[0]

players_pos_team1 = np.zeros(shape=(array_shape[0], num_players, 2))
players_pos_team2 = np.zeros(shape=(array_shape[0], num_players, 2))

for i in range(num_players):
    players_pos_team1[:,i] = data_array[:, (2*i)+1:(2*i)+3]
    players_pos_team2[:,i] = data_array[:, (2*num_players-1)+(2*i):(2*num_players-1)+(2*i)+2]

frame = 3000

in_game_team1 = players_pos_team1[frame][np.all(players_pos_team1[frame] >= 0, axis=1)]
in_game_team2 = players_pos_team2[frame][np.all(players_pos_team2[frame] >= 0, axis=1)]
tri1 = Delaunay(in_game_team1)
tri2 = Delaunay(in_game_team2)

vertex_graphs = tri2.simplices
coord = [tri2.points[x] for x in tri2.simplices]

final_points1 = []
final_points2 = []

g = Graph()

g.add_vertices(12)

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

	for points in in_game_team1:
		d0 = abs((np.cross(p1-p0, p1-points)/np.linalg.norm(p1-p0)))
		d1 = abs((np.cross(p2-p1, p2-points)/np.linalg.norm(p2-p1)))
		d2 = abs((np.cross(p0-p2, p0-points)/np.linalg.norm(p0-p2)))
		if d0 < 1.5:
			total_d0.append(d0)
		if d1 < 1.5:
			total_d1.append(d1)
		if d2 < 1.5:
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

plt.triplot(in_game_team1[:,0], in_game_team1[:,1], tri1.simplices.copy(), c='b')
plt.triplot(in_game_team2[:,0], in_game_team2[:,1], tri2.simplices.copy(), c='r')
plt.xlim(0,120)
plt.ylim(0,60)
plt.scatter(in_game_team1[:,0], in_game_team1[:,1], c='b')
plt.scatter(in_game_team2[:,0], in_game_team2[:,1], c='r')
plt.show()

for i in range(len(final_points2)):
	x1, y1 = final_points1[i]
	x2, y2 = final_points2[i]
	saida1 = [x1, x2]
	saida2 = [y1, y2]
	plt.plot(saida1, saida2, marker = 'o', c='r')
plt.show()

print(g)

