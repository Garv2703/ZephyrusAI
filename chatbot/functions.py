import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

class ChatBotModel:
    def __init__(self, data_path=os.path.join(os.path.dirname(__file__), 'Datasets/chatbot_dataset.csv')):
        self.data_path = data_path
        self.data = self._load_data()
        self.model, self.vectorizer = self._train_model()

    def _load_data(self):
        try:
            data = pd.read_csv(self.data_path)
            data = data.drop_duplicates()  # Remove duplicates
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"The data file {self.data_path} does not exist.")
        except pd.errors.EmptyDataError:
            raise ValueError("The data file is empty or cannot be read.")

    def _train_model(self):
        X = self.data['Input']
        Y = self.data['Intent']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Create a model pipeline
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(X_train, y_train)

        # Print classification report for the test set
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Return the model and vectorizer
        return model, model.named_steps['tfidfvectorizer']

    def predict_response(self, user_input):
        try:
            # Predict the intent of the user input
            intent = self.model.predict([user_input])[0]
            
            # Fetch corresponding responses from the dataset
            responses = self.data[self.data['Intent'] == intent]['Response'].unique()
            if responses.size > 0:
                # Select a random response to ensure variety
                response = responses[0]
            else:
                response = "Sorry, I don't have a response for that."

            return response
        except Exception as e:
            return f"An error occurred: {e}"
