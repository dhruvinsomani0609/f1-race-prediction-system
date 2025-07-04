"""
Simple F1 Race Prediction System
Predicts race winners and driver positions using Random Forest and FastF1 API
"""

import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

class SimpleF1Predictor:
    def __init__(self):
        """Initialize the F1 Predictor"""
        fastf1.Cache.enable_cache("f1_cache")
        self.driver_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()
        self.circuit_encoder = LabelEncoder()
        self.winner_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.position_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.data = pd.DataFrame()
        self.is_trained = False
        self.allowed_drivers_2025 = [
            'George Russell', 'Andrea Kimi Antonelli', 'Charles Leclerc', 'Lewis Hamilton',
            'Max Verstappen', 'Yuki Tsunoda', 'Lando Norris', 'Oscar Piastri',
            'Fernando Alonso', 'Lance Stroll', 'Pierre Gasly', 'Jack Doohan',
            'Esteban Ocon', 'Oliver Bearman', 'Liam Lawson', 'Alexander Albon',
            'Carlos Sainz', 'Nico Hulkenberg', 'Gabriel Bortoleto'
        ]

    def collect_race_data(self, years=[2024, 2025]):
        """Collect F1 race data for specified seasons"""
        all_race_data = []

        for year in years:
            print(f"\U0001F3CE️  Collecting F1 data for {year} season...")
            try:
                schedule = fastf1.get_event_schedule(year)
            except Exception as e:
                print(f"❌ Could not load schedule for {year}: {e}")
                continue

            latest_completed_round = 0
            for i, event in schedule.iterrows():
                if pd.Timestamp.now(tz=event['Session1Date'].tz) > event['Session3Date']:
                    latest_completed_round = event['RoundNumber']

            if latest_completed_round == 0:
                print(f"⚠️ No completed races yet for {year}. Skipping...")
                continue

            for round_num in range(1, latest_completed_round + 1):
                try:
                    race = fastf1.get_session(year, round_num, 'R')
                    race.load()
                    qualifying = fastf1.get_session(year, round_num, 'Q')
                    qualifying.load()
                    quali_results = qualifying.results
                    results = race.results
                    if results.empty:
                        continue

                    for idx, driver in results.iterrows():
                        if year == 2025 and driver['FullName'] not in self.allowed_drivers_2025:
                            continue

                        race_data = {
                            'year': year,
                            'round': round_num,
                            'circuit': race.event['Location'],
                            'driver': driver['Abbreviation'],
                            'driver_name': driver['FullName'],
                            'team': driver['TeamName'],
                            'grid_position': driver['GridPosition'],
                            'finish_position': driver['Position'],
                            'points': driver['Points'],
                            'status': driver['Status'],
                            'is_winner': 1 if driver['Position'] == 1 else 0,
                            'finished_race': 1 if driver['Status'] == 'Finished' else 0,
                        }

                        driver_quali = quali_results[quali_results['Abbreviation'] == driver['Abbreviation']]
                        if not driver_quali.empty:
                            race_data['quali_position'] = driver_quali.iloc[0]['Position']
                            q_times = [
                                driver_quali.iloc[0].get('Q3'),
                                driver_quali.iloc[0].get('Q2'),
                                driver_quali.iloc[0].get('Q1')
                            ]
                            best_time = None
                            for q_time in q_times:
                                if pd.notna(q_time):
                                    best_time = q_time.total_seconds() if hasattr(q_time, 'total_seconds') else 0
                                    break
                            race_data['quali_time'] = best_time if best_time else 0
                        else:
                            race_data['quali_position'] = 20
                            race_data['quali_time'] = 0

                        all_race_data.append(race_data)

                    print(f"✅ Processed Round {round_num} of {year}")

                except Exception as e:
                    print(f"⚠️ Error processing Round {round_num} of {year}: {str(e)}")
                    continue

        self.data = pd.DataFrame(all_race_data)
        print(f"🌟 Collected {len(self.data)} race records for {years}")
        return self.data

    def prepare_features(self):
        """Prepare data for model training"""
        df = self.data.copy()
        df = df.dropna(subset=['finish_position', 'driver', 'team', 'circuit'])

        df['driver_encoded'] = self.driver_encoder.fit_transform(df['driver'])
        df['team_encoded'] = self.team_encoder.fit_transform(df['team'])
        df['circuit_encoded'] = self.circuit_encoder.fit_transform(df['circuit'])
        df['quali_vs_grid'] = df['quali_position'] - df['grid_position']

        features = [
            'driver_encoded', 'team_encoded', 'circuit_encoded',
            'grid_position', 'quali_position', 'quali_time', 'points',
            'quali_vs_grid'
        ]

        df = df.fillna(0)
        return df, features

    def train_models(self):
        print("🚀 Training prediction models...")
        df, features = self.prepare_features()
        X = df[features]
        y_winner = df['is_winner']
        y_position = df['finish_position']

        X_train, X_test, y_winner_train, y_winner_test, y_pos_train, y_pos_test = train_test_split(
            X, y_winner, y_position, test_size=0.2, random_state=42, stratify=y_winner
        )

        self.winner_model.fit(X_train, y_winner_train)
        self.position_model.fit(X_train, y_pos_train)

        self.is_trained = True
        print("✅ Model training completed!")

    def predict_race_2025(self):
        """Make predictions for 2025 drivers using past 1-year data"""
        if not self.is_trained:
            raise ValueError("Models not trained yet! Please train models first.")

        print("\U0001F3CE️ Making 2025 Race Predictions using recent 1-year data...")

        latest_round = self.data[self.data['year'] == 2025]['round'].max()

        recent_data = self.data[
            ((self.data['year'] == 2025) & (self.data['round'] <= latest_round)) |
            (self.data['year'] == 2024)
        ].sort_values(by=['year', 'round'], ascending=[False, False])

        recent_data = recent_data[recent_data['driver_name'].isin(self.allowed_drivers_2025)]
        recent_data = recent_data.drop_duplicates(subset=['driver'])

        predictions = []
        for _, row in recent_data.iterrows():
            try:
                driver_encoded = self.driver_encoder.transform([row['driver']])[0]
                team_encoded = self.team_encoder.transform([row['team']])[0]
                circuit_encoded = self.circuit_encoder.transform([row['circuit']])[0]
            except ValueError:
                continue

            features = [
                driver_encoded,
                team_encoded,
                circuit_encoded,
                row['grid_position'],
                row['quali_position'],
                row['quali_time'],
                row['points'],
                row['quali_position'] - row['grid_position']
            ]

            win_prob = self.winner_model.predict_proba([features])[0][1]
            predicted_position = self.position_model.predict([features])[0]

            predictions.append({
                'driver': row['driver'],
                'driver_name': row['driver_name'],
                'team': row['team'],
                'win_probability': win_prob,
                'predicted_position': predicted_position,
                'grid_position': row['grid_position'],
                'quali_position': row['quali_position']
            })

        results_df = pd.DataFrame(predictions)
        results_df = results_df.sort_values('win_probability', ascending=False)
        results_df['predicted_rank'] = range(1, len(results_df) + 1)

        print("\n\U0001F3C6 2025 Race Winner Predictions:")
        print("=" * 60)
        print(results_df[['predicted_rank', 'driver_name', 'team', 'win_probability', 'predicted_position']].to_string(index=False))

        print(f"\n\U0001F947 Most Likely Winner: {results_df.iloc[0]['driver_name']} ({results_df.iloc[0]['team']}) - {results_df.iloc[0]['win_probability']:.3f}")
        print(f"\U0001F948 Second Most Likely: {results_df.iloc[1]['driver_name']} ({results_df.iloc[1]['team']}) - {results_df.iloc[1]['win_probability']:.3f}")
        print(f"\U0001F949 Third Most Likely: {results_df.iloc[2]['driver_name']} ({results_df.iloc[2]['team']}) - {results_df.iloc[2]['win_probability']:.3f}")

        return results_df

def main():
    print("🏎️  F1 Race Prediction System")
    print("=" * 50)
    predictor = SimpleF1Predictor()
    print("Starting data collection...")
    data = predictor.collect_race_data(years=[2024, 2025])

    if not data.empty:
        predictor.train_models()
        predictions = predictor.predict_race_2025()
        predictions.to_csv('f1_2025_predictions.csv', index=False)
        print("\n💾 Predictions saved to 'f1_2025_predictions.csv'")
    else:
        print("❌ No data collected. Please check your internet connection.")

if __name__ == "__main__":
    main()