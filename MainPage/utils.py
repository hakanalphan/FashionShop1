import os
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from numpy.linalg import norm

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('media', 'uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.read())
        return True
    except Exception as e:
        print(e)
        return False

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def content_based_recommendation(features, feature_list, filenames, k=5):
    # Using PCA to reduce dimensionality
    pca = PCA(n_components=100)
    reduced_features = pca.fit_transform(feature_list)
    reduced_query_features = pca.transform(features.reshape(1, -1))
    
    # Calculate cosine similarity
    similarities = cosine_similarity(reduced_query_features, reduced_features)
    
    # Get top k similar images
    indices = similarities.argsort()[0][-k:][::-1]
    return [filenames[i] for i in indices]
