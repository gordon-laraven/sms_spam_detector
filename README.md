# SMS Spam Classification with Gradio

## Table of Contents
1. [Background](#background)
2. [Files](#files)
3. [Challenge Instructions](#challenge-instructions)
   - [Create the SMS Classification Function](#create-the-sms-classification-function)
   - [Create the SMS Prediction Function](#create-the-sms-prediction-function)
   - [Create the Gradio Interface Application](#create-the-gradio-interface-application)
4. [References](#references)

## Background
The goal of this project is to refactor an existing SMS text classification solution into a function that constructs a linear Support Vector Classification (SVC) model. Once the model is created and trained, we will build a Gradio application that enables users to test text messages and receive instant feedback on whether a message is classified as spam or not based on the model's predictions.

## Files
- **Starter Code and Dataset**: Download the starter files from this link: [Module 21 Challenge Files](https://static.bc-edx.com/ai/ail-v-1-0/m21/lms/starter/M21_Starter_Code.zip). The main files included are:
  - `gradio_sms_text_classification.ipynb`
  - `sms_text_classification_solution.ipynb`
  - `SMSSpamCollection.csv`

## Challenge Instructions
The following steps outline how to implement the SMS classification functionality and build the Gradio interface for prediction.

### Create the SMS Classification Function
1. Import required libraries.
    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    import gradio as gr
    ```

2. Define the `sms_classification` function to build the SVC model. This function will:
   - Set the features variable to the text message column of the DataFrame.
   - Set the target variable to the "label" column.
   - Split the data into training and testing sets with a `test_size` of 33%.
   - Build a pipeline that includes `TfidfVectorizer` and `LinearSVC`.
   - Fit the model to the training data.
   
   Here's the implementation:
   ```python
   def sms_classification(sms_text_df):
       """
       Perform SMS classification using a pipeline with TF-IDF vectorization and Linear Support Vector Classification.

       Parameters:
       - sms_text_df (pd.DataFrame): DataFrame containing 'text_message' and 'label' columns.

       Returns:
       - text_clf (Pipeline): Fitted pipeline model.
       """
       X = sms_text_df['text_message']
       y = sms_text_df['label']
       x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

       text_clf = Pipeline([
           ('tfidf', TfidfVectorizer()),
           ('clf', LinearSVC())
       ])

       return text_clf.fit(x_train, y_train)

   # Load the dataset into a DataFrame
   sms_text_df = pd.read_csv('SMSSpamCollection.csv')
   text_clf = sms_classification(sms_text_df)
   ```

### Create the SMS Prediction Function
1. Create the `sms_prediction` function that utilizes the trained model to classify new text messages.
   - This function will take a new text message as input and return whether it's classified as "spam" or "not spam".
   
   Here’s the implementation:
   ```python
   def sms_prediction(text):
       """
       Predict whether a given text message is spam or not.

       Parameters:
       - text (str): The text message to be classified.

       Returns:
       - str: A message indicating whether the text is classified as spam or not.
       """
       prediction = text_clf.predict([text])
       if prediction[0] == 'ham':
           return f'The text message: "{text}", is not spam.'
       else:
           return f'The text message: "{text}", is spam.'
   ```

### Create the Gradio Interface Application
1. Use Gradio to create an interface that will take input from users for text messages and display the classification output.
   
   Implement the Gradio interface as follows:
   ```python
   sms_app = gr.Interface(
       fn=sms_prediction,
       inputs=[gr.Textbox(label="Add the message you want to test here")],
       outputs=[gr.Textbox(lines=5, label="The result of the prediction: ")]
   )

   # Launch the app
   sms_app.launch(show_error=True)
   ```

## References
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Gradio Documentation](https://gradio.app/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

