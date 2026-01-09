from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, roc_curve, auc)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.best_params = None

    def train_logistic_regression(self, X_train, y_train, use_grid_search=False):
        print("\nTraining Logistic Regression...")
        
        if use_grid_search:
            print("Starting Grid Search for Logistic Regression...")
            param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2']}
            grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3, verbose=1)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            self.best_params = grid.best_params_
            print(f"Best Params: {self.best_params}")
        else:
            self.model = LogisticRegression(C=1.0, max_iter=1000)
            self.model.fit(X_train, y_train)

    def train_random_forest(self, X_train, y_train, use_grid_search=False):
        print("\nTraining Random Forest...")

        if use_grid_search:
            print("Starting Grid Search for Random Forest (this may take time)...")
            # making the grid smaller for faster processing
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, verbose=1, n_jobs=-1)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            self.best_params = grid.best_params_
            print(f"Best Params: {self.best_params}")
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if not self.model:
            print("Model not trained yet.")
            return

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] # probabilities for ROC
        
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        #  visualizations
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Receiver Operating Characteristic (ROC)')
        axes[1].legend(loc="lower right")

        plt.tight_layout()
        plt.show()