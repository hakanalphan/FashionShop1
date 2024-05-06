from django.shortcuts import render
from .models import Product, UploadedImage
import numpy as np
import pickle
import os
from django.conf import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
import tensorflow as tf

# Create your views here.
def index(request):
    products = Product.objects.all()
    context = {'products': products}
    return render(request, 'index.html', context)

from .forms import UploadImageForm

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Load feature vectors and filenames
feature_list = np.array(pickle.load(open(os.path.join(settings.BASE_DIR, 'static/featurevector.pkl'), 'rb')))
filenames = pickle.load(open(os.path.join(settings.BASE_DIR, 'static/filename.pkl'), 'rb'))
image_names = [os.path.basename(file) for file in filenames]

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join(settings.MEDIA_ROOT, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.read())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.resnet50.preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def content_based_recommendation(features, feature_list, filenames, k=5):
    pca = PCA(n_components=100)
    reduced_features = pca.fit_transform(feature_list)
    reduced_query_features = pca.transform(features.reshape(1, -1))
    similarities = cosine_similarity(reduced_query_features, reduced_features)
    indices = similarities.argsort()[0][-k:][::-1]
    return [filenames[i] for i in indices]

def collaborative_filtering_recommendation(image_name, similarity_matrix, image_names, k=5):
    index = image_names.index(image_name)
    sim_scores = list(enumerate(similarity_matrix[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]
    recommended_images = [image_names[i[0]] for i in sim_scores]
    return recommended_images

def knn_recommendation(features, feature_list, filenames, k=5):
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return [filenames[i] for i in indices[0]]

def svm_recommendation(features, feature_list, filenames):
    svm = SVC(kernel='linear')
    svm.fit(feature_list, [os.path.basename(file) for file in filenames])
    prediction = svm.predict([features])
    return [filenames[np.where(np.array([os.path.basename(file) for file in filenames]) == prediction)[0][0]]]

def fashion_recommender(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['image']
            if save_uploaded_file(uploaded_file):
                # Display the uploaded file
                uploaded_image = UploadedImage(image=uploaded_file)
                uploaded_image.save()

                # Feature extraction
                features = feature_extraction(os.path.join(settings.MEDIA_ROOT, uploaded_file.name), model)

                # Content-based recommendation
                recommended_images_content_based = content_based_recommendation(features, feature_list, image_names)
                
                # Collaborative filtering recommendation
                similarity_matrix = cosine_similarity(feature_list)
                recommended_images_collaborative = collaborative_filtering_recommendation(os.path.basename(uploaded_file.name), similarity_matrix, image_names)
                
                # kNN recommendation
                recommended_images_knn = knn_recommendation(features, feature_list, filenames)
                
                # SVM recommendation
                recommended_images_svm = svm_recommendation(features, feature_list, filenames)

                return render(request, 'recommendations.html', {
                    'uploaded_file': uploaded_file,
                    'recommended_images_content_based': recommended_images_content_based,
                    'recommended_images_collaborative': recommended_images_collaborative,
                    'recommended_images_knn': recommended_images_knn,
                    'recommended_images_svm': recommended_images_svm,
                })
    else:
        form = UploadImageForm()
    return render(request, 'upload.html', {'form': form})
