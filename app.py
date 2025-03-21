from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask_cors import CORS 


app = Flask(__name__)

CORS(app)

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

@app.route('/')
def home():
    # Serve the HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()

    batting_team = data['batting_team']
    bowling_team = data['bowling_team']
    selected_city = data['city']
    target = data['target']
    score = data['score']
    overs = data['overs']
    wickets = data['wickets']

    # Prediction logic
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0  # To avoid division by zero
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0  # To avoid division by zero

    # Creating a DataFrame for the input
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict the probability
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Return the result as JSON
    return jsonify({
        'batting_team_probability': round(win * 100, 2),
        'bowling_team_probability': round(loss * 100, 2)
    })

if __name__ == '__main__':
    app.run()
