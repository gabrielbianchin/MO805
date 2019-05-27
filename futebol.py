import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import Delaunay

data_array = np.loadtxt('CapSpoT2Suav.2d').astype('int')

array_shape = data_array.shape

num_players = int((array_shape[1]-1)/4) - 1
num_frames = array_shape[0]

players_pos_team1 = np.zeros(shape=(array_shape[0], num_players, 2))
players_pos_team2 = np.zeros(shape=(array_shape[0], num_players, 2))

for i in range(num_players):
    players_pos_team1[:,i] = data_array[:, (2*i)+1:(2*i)+3]
    players_pos_team2[:,i] = data_array[:, (2*num_players-1)+(2*i):(2*num_players-1)+(2*i)+2]

#fig = plt.figure()

frame = 3000

in_game_team1 = players_pos_team1[frame][np.all(players_pos_team1[frame] >= 0, axis=1)]
in_game_team2 = players_pos_team2[frame][np.all(players_pos_team2[frame] >= 0, axis=1)]
tri1 = Delaunay(in_game_team1)
tri2 = Delaunay(in_game_team2)

coord_groups = [tri1.points[x] for x in tri1.simplices]

for conj_points in coord_groups:
	p1 = conj_points[0]
	p2 = conj_points[1]
	p3 = conj_points[2]
	for pointt in in_game_team2:
		d1 = np.cross(abs(p2-p1),abs(pointt - p1))/np.linalg.norm(abs(p2 - p1))
		d2 = np.cross(abs(p3-p2),abs(pointt - p2))/np.linalg.norm(abs(p3 - p2))
		d3 = np.cross(abs(p1-p3),abs(pointt - p3))/np.linalg.norm(abs(p1 - p3))
		if d1 < 1 or d2 < 1 or d3 < 1:
			print(conj_points)


plt.triplot(in_game_team1[:,0], in_game_team1[:,1], tri1.simplices.copy(), c='b')
plt.triplot(in_game_team2[:,0], in_game_team2[:,1], tri2.simplices.copy(), c='r')
plt.xlim(0,120)
plt.ylim(0,60)
plt.scatter(in_game_team1[:,0], in_game_team1[:,1], c='b')
plt.scatter(in_game_team2[:,0], in_game_team2[:,1], c='r')
plt.show()