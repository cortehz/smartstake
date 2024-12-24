from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.enhanced_predictor import EnhancedNBAPredictor
from data.collector import NBADataCollector
from datetime import datetime, timedelta
import pandas as pd
import json
from typing import List, Dict
import asyncio
import schedule
import threading

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Expo app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our predictor and data collector
predictor = EnhancedNBAPredictor()
data_collector = NBADataCollector()

# Load the trained model
predictor.load_model('models/trained_model.joblib')

@app.get("/")
async def root():
    return {"message": "NBA Predictor API is running"}


@app.get("/predictions/today")
async def get_todays_predictions():
    try:
        # Get today's games and player data
        today = datetime.now().strftime('%Y-%m-%d')
        games_data = data_collector.get_todays_games(today)
        
        predictions_list = []
        
        for game in games_data:
            # Prepare player data for predictions
            player_data = data_collector.get_player_features(
                game['player_id'],
                game['game_date']
            )
            
            # Get prediction and confidence intervals
            prediction, confidence = predictor.predict(pd.DataFrame([player_data]))
            
            predictions_list.append({
                "playerId": game['player_id'],
                "playerName": game['player_name'],
                "predictedPoints": round(float(prediction[0]), 1),
                "confidence": {
                    "lower": round(float(confidence['lower_bound'][0]), 1),
                    "upper": round(float(confidence['upper_bound'][0]), 1)
                },
                "opponent": game['opponent'],
                "isHome": game['is_home'],
                "gameTime": game['game_time'],
                "isBackToBack": game['is_back_to_back']
            })
        
        return predictions_list
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/players/{player_id}/history")
async def get_player_history(player_id: int):
    try:
        # Get last 20 games
        history = data_collector.get_player_game_history(player_id, limit=20)
        
        return [{
            "date": game['game_date'],
            "points": game['points'],
            "predictedPoints": game['predicted_points'],
            "opponent": game['opponent'],
            "isHome": game['is_home']
        } for game in history]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/accuracy")
async def get_model_accuracy():
    try:
        return {
            "currentMetrics": predictor.model_metrics,
            "featureImportance": predictor.feature_importance.to_dict('records'),
            "accuracyOverTime": data_collector.get_prediction_accuracy_history()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))