from futebol import *
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

data_folder = './data/'
files_ant = glob.glob(data_folder + '*.ant')
files_2d = []
for file in files_ant:
    base_name = file.split('/')[-1].split('.')[0]
    files_2d.append(data_folder + base_name + 'suav.2d')

distance=0
#classes_names = ['Domínio','Passe','Drible','Finalização-chute','Finalização-cabeca','Desarme (inf)','Desarme (sup)','Defesa Goleiro','Saida do Goleiro','Tiro-de-meta','Lateral','Escanteio','Impedimento','Falta','Gol', 'Condução']
classes_names = np.array(['defense field', 'neutral', 'danger'])
# classes_names = np.array(['Finalização-chute','Defesa Goleiro','Escanteio'])


properties = evaluate_list(files_ant, files_2d, region=classes_names, distance=0.75)
labels_properties = generate_labels(properties)

X = contatenate_properties(properties, axis=0)
y = contatenate_properties(labels_properties, axis=0)

#clf = svm.SVC(kernel='rbf')
clf = RandomForestClassifier(n_estimators=150, max_depth=25, random_state=0)

X_balanced, y_balanced = balance_data(X, y)

cross_acc, properties_relevance, acc, recall, matrix, recall = evaluate_prop(clf, X_balanced, y_balanced, classes_names)

print(cross_acc)
print()
print(properties_relevance)
print()
print(acc)
print()
print(recall)
plt.show()