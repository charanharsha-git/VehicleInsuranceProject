import pickle
import numpy as np
pickled_model_name = pickle.load(open('best_product_classification_model_name.pkl', 'rb'))

print(pickled_model_name)
pickled_model = pickle.load(open('best_product_classification_model.pkl', 'rb'))
a=pickled_model.predict([])