from sklearn.tree import export_graphviz
from six import StringIO 
import re

def create_nodo(clase_mayoritaria, informacion, left_samples, right_samples, split, class_value, negative_class='NO', positive_class='SI', umbral = 0.5, numero_total_de_muestras=1):
    separador = "<br/>"

    total_samples = left_samples + right_samples
    porcentaje_muestras_totales = round((total_samples/numero_total_de_muestras )*100, 2)
    left_probs = round(left_samples/total_samples, 2)
    right_probs = round(right_samples/total_samples, 2)
    titulo = "label=<Clase mayoritaria = "
    if (left_probs >= umbral):
        titulo = titulo + negative_class
    else:
        titulo = titulo + positive_class
        
    poblacion_total = str(porcentaje_muestras_totales)+"%"
    
    probs = "Probs=["+str(left_probs)+", "+str(right_probs)+"]"
    samples = "Samples = ["+str(total_samples)+", "+str(left_samples)+", "+str(right_samples)+"]"

    new_string = titulo+ "<br/>" +poblacion_total + "<br/>" +probs+ "<br/>" +samples+ "<br/>"+ informacion 

    if (split != ""):
        new_string = new_string + "<br/>" + split

    new_string = new_string +">," 

    if (left_probs < umbral):
        new_string = new_string + "fillcolor=\"#FF6B6B\""
    else:
        new_string = new_string + "fillcolor=\"#4D96FF\""

    return new_string

def crear_arbol(clf, negative_class = 'NO', positive_class='SI', umbral = 0.5, max_depth=None):
    numero_total_de_muestras = clf.tree_.n_node_samples[0]
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = clf.feature_names_in_ ,class_names=clf.classes_, max_depth=max_depth)
    
    dot_data_value = dot_data.getvalue()
    
    result = re.findall('\[(label\=<(.*)<br/>(.*)=(.*)<br/>(.*)=(.*)<br/>(.*)=\s\[(\d+),\s(\d+)\]<br/>(.*)=(.*)\>,(.*))]', dot_data_value, re.IGNORECASE)

    for r in result:
        i_todo = 0                        #0 todo
        i_titulo = 1                      #1 titulo
        i_information_label = 2           #2 information label
        i_information_value = 3           #3 information value
        i_samples_label = 4               #4 samples label
        i_samples_values = 5              #5 samples values
        i_leaves_label = 6                #6 leaves label
        i_leaves_values_left = 7          #7 leaves values left
        i_leaves_values_right = 8         #8 leaves values right
        i_class_label = 9                 #9 class label
        i_class_value = 10                #10 class values
        i_resto = 11                      #11 resto

        condicion = r[i_titulo]

        left_samples =  int(r[i_leaves_values_left])
        right_samples =  int(r[i_leaves_values_right])

        information = r[i_information_label] + "=" + r[i_information_value]

        new_string = create_nodo(r[i_class_value].strip(), information, left_samples, right_samples, condicion, r[i_class_value].strip(), negative_class, positive_class, umbral, numero_total_de_muestras)

        dot_data_value = dot_data_value.replace(r[0], new_string)

    result = re.findall('\[(label\=<(.*)=(.*)<br/>(.*)=(.*)<br/>(.*)=\s\[(\d+),\s(\d+)\]<br/>(.*)=(.*)\>,(.*))]', dot_data_value, re.IGNORECASE)

    for r in result:
        i_todo = 0                        #0 todo
        i_information_label = 1           #1 information label
        i_information_value = 2           #2 information value
        i_samples_label = 3               #3 samples label
        i_samples_values = 4              #4 samples values
        i_leaves_label = 5                #5 leaves label
        i_leaves_values_left = 6          #6 leaves values left
        i_leaves_values_right = 7         #7 leaves values right
        i_class_label = 8                 #8 class label
        i_class_value = 9                 #9 class values
        i_resto = 10                      #10 resto

        left_samples =  int(r[i_leaves_values_left])
        right_samples =  int(r[i_leaves_values_right])

        information = r[i_information_label] + "=" + r[i_information_value]

        new_string = create_nodo(r[i_class_value].strip(), information, left_samples, right_samples, "", r[i_class_value].strip(), negative_class, positive_class, umbral, numero_total_de_muestras)

        dot_data_value = dot_data_value.replace(r[0], new_string)
        
    return dot_data_value