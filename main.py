import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer

def main():
    data_path = os.path.join('data', 'IMDB Dataset.csv')
    
    prep = DataPreprocessor(data_path)
    try:
        prep.load_data()
    except FileNotFoundError:
        print(f"Error. File not found at {data_path}")
        return

   # preprocessing and feature engineering
    prep.preprocess_data()
    
    # EDA
    prep.show_eda()
    
    # vectorization and train-test split
    X_train, X_test, y_train, y_test = prep.split_and_vectorize()
    
    trainer = ModelTrainer()
    
    # fine tuning logistic regression
    print("\n--- Logistic Regression ---")
    trainer.train_logistic_regression(X_train, y_train, use_grid_search=True)
    trainer.evaluate(X_test, y_test) 
    
    # random forest with hyperparameter tuning
    print("\nWanna start Random Forest with Hyperparameter Tuning? (y/n)")
    ans = input()
    if ans.lower() == 'y':
        trainer.train_random_forest(X_train, y_train, use_grid_search=True)
        trainer.evaluate(X_test, y_test)

    
    print("\n" + "="*40)
    print("Write your own review to predict sentiment!")
    print("Write 'exit' to exit.")
    print("="*40)

    while True:
        user_input = input("\nEnter review: ")
        if user_input.lower() == 'exit':
            break

        # cleaning
        cleaned = prep.clean_text(user_input)
        # transform
        vectorized = prep.vectorizer.transform([cleaned])
        # predict
        prediction = trainer.model.predict(vectorized)[0]
        proba = trainer.model.predict_proba(vectorized)[0]
        
        label = "POSITIVE" if prediction == 1 else "NEGATIVE"
        confidence = proba[prediction] * 100

        print(f"Result: {label} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    main()