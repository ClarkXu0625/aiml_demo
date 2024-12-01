from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

## get a pretrained random forest classifier model to make the local server work
# Load dataset and train a model
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'random_forest_iris.pkl')
