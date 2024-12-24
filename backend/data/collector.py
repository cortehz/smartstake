import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamefinder, boxscoreadvancedv2, playerprofilev2
from datetime import datetime, timedelta
import sqlite3

class NBADataCollector:
    def __init__(self, db_path='nba_data.db'):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create necessary tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                date TEXT,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_team_score INTEGER,
                away_team_score INTEGER,
                season INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_stats (
                game_id TEXT,
                player_id INTEGER,
                minutes REAL,
                points INTEGER,
                rebounds INTEGER,
                assists INTEGER,
                steals INTEGER,
                blocks INTEGER,
                turnovers INTEGER,
                fg_pct REAL,
                fg3_pct REAL,
                ft_pct REAL,
                plus_minus INTEGER,
                FOREIGN KEY (game_id) REFERENCES games (game_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_info (
                player_id INTEGER PRIMARY KEY,
                name TEXT,
                position TEXT,
                team_id INTEGER,
                height TEXT,
                weight INTEGER
            )
        ''')

        conn.commit()
        conn.close()

    def collect_historical_data(self, start_date, end_date):
        # Convert dates to proper format
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        # Get games
        games = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=start_date,
            date_to_nullable=end_date
        ).get_data_frames()[0]

        conn = sqlite3.connect(self.db_path)
        
        for _, game in games.iterrows():
            # Store game data
            game_data = {
                'game_id': game['GAME_ID'],
                'date': game['GAME_DATE'],
                'home_team_id': game['TEAM_ID'] if game['MATCHUP'].split()[-1] == 'vs.' else game['OPPONENT_TEAM_ID'],
                'away_team_id': game['OPPONENT_TEAM_ID'] if game['MATCHUP'].split()[-1] == 'vs.' else game['TEAM_ID'],
                'home_team_score': game['PTS'] if game['MATCHUP'].split()[-1] == 'vs.' else game['OPP_PTS'],
                'away_team_score': game['OPP_PTS'] if game['MATCHUP'].split()[-1] == 'vs.' else game['PTS'],
                'season': game['SEASON_ID']
            }

            # Get detailed box score
            box_score = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game['GAME_ID']).get_data_frames()[0]
            
            # Store player stats
            for _, player_stats in box_score.iterrows():
                stats_data = {
                    'game_id': game['GAME_ID'],
                    'player_id': player_stats['PLAYER_ID'],
                    'minutes': player_stats['MIN'],
                    'points': player_stats['PTS'],
                    'rebounds': player_stats['REB'],
                    'assists': player_stats['AST'],
                    'steals': player_stats['STL'],
                    'blocks': player_stats['BLK'],
                    'turnovers': player_stats['TOV'],
                    'fg_pct': player_stats['FG_PCT'],
                    'fg3_pct': player_stats['FG3_PCT'],
                    'ft_pct': player_stats['FT_PCT'],
                    'plus_minus': player_stats['PLUS_MINUS']
                }
                
                self.store_data(conn, 'player_stats', stats_data)

            self.store_data(conn, 'games', game_data)

        conn.close()

    def store_data(self, conn, table, data):
        cursor = conn.cursor()
        placeholders = ', '.join(['?' for _ in data])
        columns = ', '.join(data.keys())
        sql = f'INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})'
        cursor.execute(sql, list(data.values()))
        conn.commit()

    def get_player_features(self, player_id, game_date):
        conn = sqlite3.connect(self.db_path)
        
        # Get last 5 games stats
        query = '''
            SELECT ps.*
            FROM player_stats ps
            JOIN games g ON ps.game_id = g.game_id
            WHERE ps.player_id = ?
            AND g.date < ?
            ORDER BY g.date DESC
            LIMIT 5
        '''
        
        last_5_games = pd.read_sql_query(query, conn, params=[player_id, game_date])
        
        features = {
            'pts_last_5': last_5_games['points'].mean(),
            'reb_last_5': last_5_games['rebounds'].mean(),
            'ast_last_5': last_5_games['assists'].mean(),
            'min_last_5': last_5_games['minutes'].mean(),
            'plus_minus_last_5': last_5_games['plus_minus'].mean()
        }
        
        conn.close()
        return features