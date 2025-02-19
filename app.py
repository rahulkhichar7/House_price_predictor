import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained pipeline
model = pickle.load(open('house_price_pipeline.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()  # Extract JSON data
        SqFt = int(data['data'].get('SqFt', 0))
        Bedrooms = int(data['data'].get('Bedrooms', 0))
        Bathrooms = int(data['data'].get('Bathrooms', 0))
        Offers = int(data['data'].get('Offers', 0))
        Brick = data['data'].get('Brick', 'No')
        Neighborhood = data['data'].get('Neighborhood', 'East')

        # print("This is the data from the form:\n")
        # print(SqFt, '\n', Bedrooms, '\n', Bathrooms, '\n', Offers, '\n', Brick, '\n', Neighborhood, '\n')

        # Convert input into DataFrame
        data_unseen = pd.DataFrame([[SqFt, Bedrooms, Bathrooms, Offers, Brick, Neighborhood]], 
                                   columns=["SqFt", "Bedrooms", "Bathrooms", "Offers", "Brick", "Neighborhood"])
        print("This is the data_unseen:\n")
        print(data_unseen.head())

        # Make prediction
        prediction = model.predict(data_unseen)
        # print(prediction)

        return jsonify({'prediction': int(prediction[0])})  

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error details for debugging


if __name__ == '__main__':
    app.run(debug=True)
