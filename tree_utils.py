import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def giniindex(class_1, class_2):
    return 1 - ((class_1/(class_1+class_2))**2+(class_2/(class_1+class_2))**2)


###############################################
def get_instancias_nodo(inner_tree, indice):
    nro_instancias_classe_a = inner_tree.value[indice][0][0]
    nro_instancias_classe_b = inner_tree.value[indice][0][1]    
    nro_total_instancias = nro_instancias_classe_a + nro_instancias_classe_b
    return [nro_total_instancias, nro_instancias_classe_a, nro_instancias_classe_b]

def get_instancias_nodo_izquierdo(inner_tree, indice):
    return get_instancias_nodo(inner_tree, inner_tree.children_left[indice])

def get_instancias_nodo_derecho(inner_tree, indice):
    return get_instancias_nodo(inner_tree, inner_tree.children_right[indice])

def get_giniindex_nodo(inner_tree, indice):
    nro_total_instancias, nro_instancias_classe_a, nro_instancias_classe_b = get_instancias_nodo(inner_tree, indice)
    return 1 - ((nro_instancias_classe_a/nro_total_instancias)**2+(nro_instancias_classe_b/nro_total_instancias)**2)

def get_entropia_nodo(inner_tree, indice):
    return get_giniindex_nodo(inner_tree, indice)

def get_entropia_nodo_derecho (inner_tree, indice):
    return get_entropia_nodo(inner_tree, inner_tree.children_right[indice])

def get_entropia_nodo_izquierdo (inner_tree, indice):
    return get_entropia_nodo(inner_tree, inner_tree.children_left[indice])

def get_ganancia_informacion_nodo(inner_tree, indice):
    total_instancias, clase_a, clase_b = get_instancias_nodo(inner_tree, indice)
    izq_total_instancias, izq_clase_a, izq_clase_b = get_instancias_nodo_izquierdo(inner_tree, indice)
    der_total_instancias, der_clase_a, der_clase_b = get_instancias_nodo_derecho(inner_tree, indice)
    
    entropia_nodos_hijos = (
        izq_total_instancias / total_instancias * get_entropia_nodo_izquierdo(inner_tree, indice) 
        + der_total_instancias / total_instancias * get_entropia_nodo_derecho(inner_tree, indice)
    )
    
    return get_entropia_nodo(inner_tree, indice) - entropia_nodos_hijos    

def get_clase_nodo_validando_umbral (inner_tree, indice, umbral, nombre_clase_negativa, nombre_clase_positiva):
    nro_instancias_classe_a, nro_instancias_classe_b = get_instancias_nodo(inner_tree, indice)
    prob_a = nro_instancias_classe_a / (nro_instancias_classe_a + nro_instancias_classe_b)
    prob_b = 1 - prob_a
    if(prob_a > umbral):
        return nombre_clase_negativa
    else:
        return nombre_clase_positiva
    
def get_informacion_de_nodo(inner_tree, indice):
    entropia_nodo = get_entropia_nodo(inner_tree, indice)
    entropia_nodo_izquierdo = get_entropia_nodo_izquierdo (inner_tree, indice)
    entropia_nodo_derecho = get_entropia_nodo_derecho (inner_tree, indice)
    
    samples_nodo, samples_nodo_clase_a, samples_nodo_clase_b = get_instancias_nodo(inner_tree, indice)
    samples_nodo_izq, samples_nodo_izq_clase_a, samples_nodo_der_clase_b = get_instancias_nodo_izquierdo(inner_tree, indice)
    samples_nodo_der, samples_nodo_der_clase_a, samples_nodo_der_clase_b = get_instancias_nodo_derecho(inner_tree, indice)
    
    proporcion_samples_nodo_izquierdo = samples_nodo_izq / samples_nodo
    proporcion_samples_nodo_derecho = samples_nodo_der / samples_nodo
    
    entropia_nodo_izquierdo = get_entropia_nodo_izquierdo(inner_tree, indice)
    entropia_nodo_derecho = get_entropia_nodo_derecho(inner_tree, indice)
    
    entropia_nodo = get_entropia_nodo(inner_tree, indice)
    
    information_gain = get_ganancia_informacion_nodo(inner_tree, indice)
    
    return { 
        'samples_nodo': int(samples_nodo),
        'samples_nodo_izquierdo': int(samples_nodo_izq),
        'samples_nodo_derecho': int(samples_nodo_der),
        
        'proporcion_samples_nodo_izquierdo': round(proporcion_samples_nodo_izquierdo, 2),
        'proporcion_samples_nodo_derecho': round(proporcion_samples_nodo_derecho, 2) ,
        
        'entropia_nodo': round(entropia_nodo, 4),
        'entropia_nodo_izquierdo': round(entropia_nodo_izquierdo, 4),
        'entropia_nodo_derecho': round(entropia_nodo_derecho, 4),
        
        'information_gain': round(information_gain, 4)
    }

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def calcular_metricas(y_true, y_pred, labels): #labels: array de string, la clase negativa va primero
    [tn, fp, fn, tp] = confusion_matrix(y_true, y_pred,  labels=labels).ravel()
    
    accuracy = safe_div((tp+tn) , (tp + tn + fp + fn))
    
    recall = safe_div(tp , tp + fn)
    
    precision = safe_div(tp , tp + fp)
    
    specificity = safe_div(tn , tn + fp)
    
    tasa_falso_positivo = safe_div(fp , tn + fp)
    
    tasa_falso_negativo = safe_div(fn , tn + fp)
    
    f1 = 2 * safe_div( precision*recall , precision+recall)
    
    metrica_negocio = tp * 4000 - fp 
    
    metrica_negocio = ((4000*tp) - (200*fp))/(tp + fn)
    metrica_negocio_optima = (4000*(tp+fn))/(tp + fn)
    
    return {
        'true negatives':tn, 
        'true positives':tp, 
        'false positives': fp, 
        'false negatives': fn,
        'accuracy': round(accuracy, 4),
        'recall': round(recall, 4),
        'precision': round(precision, 4),
        'specificity': round(specificity, 4),
        'tasa falso positivo': round(tasa_falso_positivo, 4),
        'tasa falso negativo': round(tasa_falso_negativo, 4),
        'f1': round(f1, 4),
        'metrica negocio': metrica_negocio,
        'metrica optima negocio': metrica_negocio_optima
    }

def evaluar_corte(array_prob_clase_negativa, umbral_corte, nombre_clase_positiva, nombre_clase_negativa):
    return np.where(array_prob_clase_negativa <= umbral_corte, nombre_clase_positiva, nombre_clase_negativa)

###############################################



def entropia_de_nodo (df_tree, idx):
    nodo_raiz_class_1_samples = df_tree.loc[idx, :]['NO']
    nodo_raiz_class_2_samples = df_tree.loc[idx, :]['SI']
    return giniindex(nodo_raiz_class_1_samples, nodo_raiz_class_2_samples)

def entropia_de_nodo_izquierdo(df_tree, idx_parent):
    return entropia_de_nodo(df_tree, df_tree.loc[idx_parent, :]['children_left'])

def entropia_de_nodo_derecho(df_tree, idx_parent):
    return entropia_de_nodo(df_tree, df_tree.loc[idx_parent, :]['children_right'])

def samples_de_nodo(df_tree, idx):
    return df_tree.loc[idx, :]['node_samples']

def samples_de_nodo_izquierdo(df_tree, idx_parent):
    return samples_de_nodo(df_tree, df_tree.loc[idx_parent, :]['children_left'])

def samples_de_nodo_derecho(df_tree, idx_parent):
    return samples_de_nodo(df_tree, df_tree.loc[idx_parent, :]['children_right'])

def get_info_node(df_tree, idx):
    entropia_nodo = entropia_de_nodo (df_tree, idx)
    entropia_nodo_izquierdo = entropia_de_nodo_izquierdo (df_tree, idx)
    entropia_nodo_derecho = entropia_de_nodo_derecho (df_tree, idx)
    
    samples_nodo = samples_de_nodo(df_tree, idx)
    samples_nodo_izquierdo = samples_de_nodo_izquierdo(df_tree, idx)
    samples_nodo_derecho = samples_de_nodo_derecho(df_tree, idx)
    
    proporcion_samples_nodo_izquierdo = samples_nodo_izquierdo / samples_nodo
    proporcion_samples_nodo_derecho = samples_nodo_derecho / samples_nodo
    
    entropia_nodos_hijos = proporcion_samples_nodo_izquierdo * entropia_nodo_izquierdo + proporcion_samples_nodo_derecho * entropia_nodo_derecho
    
    information_gain = entropia_nodo - entropia_nodos_hijos
    
    return { 
        'samples_nodo': int(samples_nodo),
        'samples_nodo_izquierdo': int(samples_nodo_izquierdo),
        'samples_nodo_derecho': int(samples_nodo_derecho),
        
        'proporcion_samples_nodo_izquierdo': round(proporcion_samples_nodo_izquierdo, 2),
        'proporcion_samples_nodo_derecho': round(proporcion_samples_nodo_derecho, 2) ,
        
        'entropia_nodo': round(entropia_nodo, 4),
        'entropia_nodo_izquierdo': round(entropia_nodo_izquierdo, 4),
        'entropia_nodo_derecho': round(entropia_nodo_derecho, 4),
        'entropia_nodos_hijos': round(entropia_nodos_hijos, 4),
        
        'information_gain': round(information_gain, 4)
    }

#######################################################################################
from sklearn.tree._tree import TREE_LEAF


def get_node_class (inner_tree, index, umbral, negative_class, positive_class):
    class_a = inner_tree.value[index][0][0]
    class_b = inner_tree.value[index][0][1]
    prob_a = class_a / (class_a + class_b)
    prob_b = 1 - prob_a
    if(prob_a > umbral):
        return negative_class
    else:
        return positive_class
    
def get_node_classes (inner_tree, index, umbral, negative_class, positive_class):
    result = []
    
    if(inner_tree.children_left[index] == TREE_LEAF):
        result = [get_node_class(inner_tree, index, umbral, negative_class, positive_class)]
    else:
        result = (
            get_node_classes(inner_tree, inner_tree.children_left[index], umbral, negative_class, positive_class) +
            get_node_classes(inner_tree, inner_tree.children_right[index], umbral, negative_class, positive_class)
        )
        
    return result



def prune_index(inner_tree, index, umbral):
    clases_hojas = get_node_classes(inner_tree, index, umbral, 'NO', 'SI')
    #print(index, clases_hojas)
    res = np.array(clases_hojas) 
    unique_res = np.unique(res)
    if(len(unique_res) == 1):
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        
        #right_class = get_right_node_class(inner_tree, index, umbral, 'NO', 'SI')
        #if(left_class == right_class):
        #    inner_tree.children_left[index] = TREE_LEAF
        #    inner_tree.children_right[index] = TREE_LEAF
    
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], umbral)
        prune_index(inner_tree, inner_tree.children_right[index], umbral)

