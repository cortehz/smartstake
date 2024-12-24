import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

class EnhancedNBAPredictor:
    def __init__(self):
        self.model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=7,
            num_leaves=31,
            feature_fraction=0.8,
            subsample=0.8
        )
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.model_metrics = {}
        
    def advanced_feature_engineering(self, df):
        # Basic features from previous implementation
        df['is_home'] = df['is_home'].astype(int)
        df['back_to_back'] = df['days_rest'] == 0
        
        # Advanced rolling statistics
        rolling_windows = [5, 10, 20]
        for window in rolling_windows:
            df[f'pts_last_{window}'] = df.groupby('player_id')['points'].rolling(window).mean().reset_index(0, drop=True)
            df[f'pts_std_{window}'] = df.groupby('player_id')['points'].rolling(window).std().reset_index(0, drop=True)
            df[f'usage_last_{window}'] = df.groupby('player_id')['usage_rate'].rolling(window).mean().reset_index(0, drop=True)
        
        # Team pace adjustment
        df['team_pace_factor'] = df['team_pace'] / df['league_avg_pace']
        
        # Matchup specific features
        df['vs_team_defense_rating'] = df['opponent_defensive_rating']
        df['vs_position_pts_allowed'] = df.groupby('opponent_position')['points'].transform('mean')
        
        # Rest and schedule features
        df['days_since_last_game'] = df.groupby('player_id')['game_date'].diff().dt.days
        df['games_in_last_7_days'] = df.groupby('player_id').rolling('7D')['game_id'].count().reset_index(0, drop=True)
        
        # Season progress feature
        df['games_played_this_season'] = df.groupby(['player_id', 'season'])['game_id'].cumcount()
        
        # Home/Away splits with recency
        recent_home_games = df[df['is_home'] == 1].groupby('player_id').tail(10)
        recent_away_games = df[df['is_home'] == 0].groupby('player_id').tail(10)
        
        df['recent_home_avg'] = df['player_id'].map(recent_home_games.groupby('player_id')['points'].mean())
        df['recent_away_avg'] = df['player_id'].map(recent_away_games.groupby('player_id')['points'].mean())
        
        # Vegas odds features
        if 'vegas_line' in df.columns and 'over_under' in df.columns:
            df['implied_team_score'] = (df['over_under'] / 2) + (df['vegas_line'] / 2)
        
        return df

    def train(self, training_data, validation_data=None):
        # Feature engineering
        processed_train = self.advanced_feature_engineering(training_data)
        
        if validation_data is not None:
            processed_val = self.advanced_feature_engineering(validation_data)
        
        # Define features
        self.features = [col for col in processed_train.columns 
                        if col not in ['points', 'player_id', 'game_id', 'game_date']]
        
        X_train = processed_train[self.features]
        y_train = processed_train['points']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if validation_data is not None:
            X_val = processed_val[self.features]
            y_val = processed_val['points']
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train with validation set
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric='mae',
                early_stopping_rounds=50,
                verbose=100
            )
        else:
            # Train without validation set
            self.model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate training metrics
        train_predictions = self.model.predict(X_train_scaled)
        self.model_metrics['train_mse'] = mean_squared_error(y_train, train_predictions)
        self.model_metrics['train_mae'] = mean_absolute_error(y_train, train_predictions)
        
        if validation_data is not None:
            val_predictions = self.model.predict(X_val_scaled)
            self.model_metrics['val_mse'] = mean_squared_error(y_val, val_predictions)
            self.model_metrics['val_mae'] = mean_absolute_error(y_val, val_predictions)
    
    def predict(self, player_data):
        processed_data = self.advanced_feature_engineering(player_data)
        X = processed_data[self.features]
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        
        # Calculate prediction intervals (simple approach)
        prediction_std = np.std(predictions)
        confidence_intervals = {
            'lower_bound': predictions - 1.96 * prediction_std,
            'upper_bound': predictions + 1.96 * prediction_std
        }
        
        return predictions, confidence_intervals
    
    def save_model(self, path):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'feature_importance': self.feature_importance,
            'metrics': self.model_metrics
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path):
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.features = model_data['features']
        self.feature_importance = model_data['feature_importance']
        self.model_metrics = model_data['metrics']