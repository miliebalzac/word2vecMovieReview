import streamlit as st

st.title("Modèle Word2Vec")

embedding_dim = 300

#==========================================================================
# ajouté par Emilie avec l'aide de IA pour récupérer vocab_size, word2idx et idx2word  dans le fichier .py de streamlit
import pickle
with open("cours_streamlit/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

word2idx = tokenizer.word_index
idx2word = tokenizer.index_word
vocab_size = tokenizer.num_words

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# modifié par Emilie avec laide de lIA. Comme on a sauvegardé le modèle, ça ne sert à rien de le reconstituer à la main
# il faut mieux se contenter de recherger le modèle

#model = Sequential()
#model.add(Embedding(vocab_size, embedding_dim))
#model.add(GlobalAveragePooling1D())
#model.add(Dense(vocab_size, activation='softmax'))
#model.load_weights("cours_streamlit/word2vec.h5")


import tensorflow as tf

model = tf.keras.models.load_model("cours_streamlit/word2vec.h5", compile=False)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# La similitude est une métrique mesurant la distance entre deux mots. Cette distance représente la façon dont les mots sont liés entre eux.
# (k) Ajouter le code suivant pour extraire la matrice d'embeddings et définir les fonctions de similitude.

vectors = model.layers[0].trainable_weights[0].numpy()

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# modifié par Emilie avec laide de lIA. 
# on ne garde que les mots qui ont réellement un vecteur dans la matrice
word2idx = {w: i for w, i in tokenizer.word_index.items() if i < vectors.shape[0]}
idx2word = {i: w for w, i in word2idx.items()}
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import numpy as np
from sklearn.preprocessing import Normalizer

def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def print_closest(word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        st.write(idx2word[int(index_word[1])], " -- ", float(index_word[0]))

#(l) Créer des widgets Streamlit permettant à l'utilisateur d'afficher les 10 mots les plus proches d'un mot choisi, grâce à la fonction print_closest définie précédemment.
#Remarque : Une idée pour rendre le Streamlit encore plus intéractif pourrait être de laisser à l'utilisateur le choix du nombre de mots proches.

#Exemple d'utilisation de la fonction print_closest
mot = st.text_input("Entrez un mot", value="zombie").strip().lower()
nombre = st.slider("Nombre de mots proches", 1, 30, 10)

if st.button("Afficher les mots les plus proches"):
    if not mot:
        st.warning("Veuillez saisir un mot.")
    elif mot in word2idx:
        st.subheader(f"Mots les plus proches de « {mot} »")
        print_closest(mot, nombre)
    elif mot in tokenizer.word_index:
        st.warning(
            f"« {mot} » est présent dans le corpus mais trop rare : "
            f"il ne fait pas partie des 9 999 mots les plus fréquents retenus "
            f"pour l'entraînement (num_words=10000)."
        )
    else:
        st.error(f"Je ne connais pas le mot « {mot} ».")
