import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

def select_features_boruta(X, y, max_iter=100, random_state=42):
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', max_iter=max_iter, random_state=random_state)
    boruta_selector.fit(X, y)
    selected_features = np.where(boruta_selector.support_)[0]
    feature_importances = boruta_selector.ranking_
    return selected_features, feature_importances
