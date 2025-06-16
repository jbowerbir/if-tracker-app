import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, time, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict, List
import math
import json

class IFTracker:
    def __init__(self):
        # Initialize session state for data storage
        if 'entries' not in st.session_state:
            st.session_state.entries = pd.DataFrame()
        
        # Initialize formula parameters with defaults
        if 'formula_params' not in st.session_state:
            st.session_state.formula_params = {
                'fasting_linear': {'slope': 12.5, 'intercept': 12},
                'fasting_tiered': {
                    'thresholds': [12, 13, 14, 15, 16, 17, 18, 19, 20],
                    'scores': [0, 20, 50, 70, 80, 90, 92, 95, 98, 100]
                },
                'fasting_logistic': {'steepness': 0.8, 'midpoint': 16, 'max_score': 100},
                'window_linear': {'slope': -12.5, 'intercept': 12},
                'window_tiered': {
                    'thresholds': [4, 5, 6, 7, 8, 9, 10, 11, 12],
                    'scores': [100, 95, 95, 92, 90, 80, 70, 50, 20, 0]
                },
                'window_logistic': {'steepness': 1.3, 'midpoint': 7.5, 'max_score': 100},
                'midpoint_optimal': {'start': 11.0, 'end': 13.0},
                'consistency_thresholds': [0.49, 0.99, 1.49, 1.99],
                'consistency_penalties': [0, -2, -5, -10, -15],
                'discipline_criteria': {
                    'peak': {'fasting': 18, 'window': 6, 'midpoint_start': 11.0, 'midpoint_end': 13.0, 'bonus': 5},
                    'high': {'fasting': 17, 'window': 7, 'midpoint_start': 10.5, 'midpoint_end': 13.5, 'bonus': 3},
                    'good': {'fasting': 16, 'window': 8, 'midpoint_start': 10.0, 'midpoint_end': 14.0, 'bonus': 1}
                }
            }
        
        # Initialize behavior trackers with defaults
        if 'behavior_trackers' not in st.session_state:
            st.session_state.behavior_trackers = {
                'wod_strength': {'name': 'WOD - Strength', 'type': 'binary', 'active': True},
                'wod_metcon': {'name': 'WOD - MetCon', 'type': 'binary', 'active': True},
                'martial_arts': {'name': 'Martial Arts', 'type': 'count', 'active': True, 'max_value': 3},
                'yoga': {'name': 'Yoga', 'type': 'binary', 'active': True},
                'meditation': {'name': 'Meditation', 'type': 'duration', 'active': True, 'unit': 'minutes'},
                'breathing': {'name': 'Breathing Exercise', 'type': 'binary', 'active': True},
                'pt': {'name': 'Physical Therapy', 'type': 'binary', 'active': True}
            }
        
        # Initialize subjective metrics
        if 'subjective_metrics' not in st.session_state:
            st.session_state.subjective_metrics = {
                'energy_morning': {'name': 'Morning Energy', 'scale': [1, 10], 'active': True},
                'energy_afternoon': {'name': 'Afternoon Energy', 'scale': [1, 10], 'active': True},
                'energy_evening': {'name': 'Evening Energy', 'scale': [1, 10], 'active': True},
                'hunger_level': {'name': 'Overall Hunger', 'scale': [1, 10], 'active': True},
                'mood': {'name': 'Overall Mood', 'scale': [1, 10], 'active': True},
                'focus': {'name': 'Mental Focus', 'scale': [1, 10], 'active': True},
                'satisfaction': {'name': 'Diet Satisfaction', 'scale': [1, 10], 'active': True},
                'cravings': {'name': 'Cravings Intensity', 'scale': [1, 10], 'active': True}
            }
        
        # Scoring system configurations
        self.scoring_systems = {
            'linear': 'Smooth Linear Progression',
            'tiered': 'Tiered Non-Linear (Goal-Oriented)', 
            'logarithmic': 'Logarithmic Curve (Diminishing Returns)'
        }
        
        # Explanations for scoring systems
        self.explanations = {
            'fasting_duration': {
                'description': 'Length of your fasting window',
                'optimal_range': '16-20 hours',
                'benefits': 'Promotes autophagy, insulin sensitivity, and metabolic flexibility'
            },
            'eating_window': {
                'description': 'Duration of your eating window', 
                'optimal_range': '4-8 hours',
                'benefits': 'Shorter windows enhance metabolic benefits and circadian alignment'
            },
            'eating_midpoint': {
                'description': 'Middle point of your eating window',
                'optimal_range': '11:00-13:00',
                'benefits': 'Aligns with natural circadian rhythms and cortisol patterns'
            },
            'consistency': {
                'description': 'Standard deviation of eating midpoint timing',
                'optimal_range': '< 0.5 hours',
                'benefits': 'Consistent timing reinforces circadian rhythms and hormone cycles'
            }
        }

    def get_active_columns(self) -> List[str]:
        """Get list of active columns for the dataframe."""
        base_cols = ['date', 'first_meal_time', 'last_meal_time', 'sleep_time']
        
        # Add active behavior trackers
        for key, tracker in st.session_state.behavior_trackers.items():
            if tracker['active']:
                base_cols.append(key)
        
        # Add active subjective metrics
        for key, metric in st.session_state.subjective_metrics.items():
            if metric['active']:
                base_cols.append(key)
                
        return base_cols

    def ensure_dataframe_columns(self):
        """Ensure dataframe has all necessary columns."""
        required_cols = self.get_active_columns()
        
        if st.session_state.entries.empty:
            st.session_state.entries = pd.DataFrame(columns=required_cols)
        else:
            # Add missing columns
            for col in required_cols:
                if col not in st.session_state.entries.columns:
                    st.session_state.entries[col] = None

    def calculate_time_diff_hours(self, start_time: time, end_time: time) -> float:
        """Calculate difference between two times, handling overnight periods."""
        start_minutes = start_time.hour * 60 + start_time.minute
        end_minutes = end_time.hour * 60 + end_time.minute
        
        if end_minutes < start_minutes:  # Overnight
            end_minutes += 24 * 60
            
        return (end_minutes - start_minutes) / 60

    def calculate_fasting_duration(self, prev_last_meal: Optional[time], 
                                 current_first_meal: Optional[time]) -> Optional[float]:
        """Calculate fasting duration from previous day's last meal to current first meal."""
        if prev_last_meal is None or current_first_meal is None:
            return None
        return self.calculate_time_diff_hours(prev_last_meal, current_first_meal)

    def calculate_eating_window(self, first_meal: Optional[time], 
                              last_meal: Optional[time]) -> Optional[float]:
        """Calculate eating window duration."""
        if first_meal is None or last_meal is None:
            return None
        return self.calculate_time_diff_hours(first_meal, last_meal)

    def calculate_eating_midpoint(self, first_meal: Optional[time], 
                                last_meal: Optional[time]) -> Optional[float]:
        """Calculate midpoint of eating window as decimal hours from midnight."""
        if first_meal is None or last_meal is None:
            return None
            
        first_decimal = first_meal.hour + first_meal.minute / 60
        last_decimal = last_meal.hour + last_meal.minute / 60
        
        if last_decimal < first_decimal:  # Overnight eating
            last_decimal += 24
            
        midpoint = (first_decimal + last_decimal) / 2
        return midpoint % 24  # Normalize to 0-24 range

    def score_fasting_duration(self, hours: Optional[float], method: str) -> Optional[float]:
        """Score fasting duration using selected method and custom parameters."""
        if hours is None:
            return None
            
        params = st.session_state.formula_params
        
        if method == 'linear':
            p = params['fasting_linear']
            return max(0, min(100, (hours - p['intercept']) * p['slope']))
        elif method == 'tiered':
            p = params['fasting_tiered']
            for i, threshold in enumerate(p['thresholds']):
                if hours < threshold:
                    return p['scores'][i] if i > 0 else 0
            return p['scores'][-1]  # Max score if above all thresholds
        elif method == 'logarithmic':
            p = params['fasting_logistic']
            return round(p['max_score'] / (1 + math.exp(-p['steepness'] * (hours - p['midpoint']))), 1)

    def score_eating_window(self, hours: Optional[float], method: str) -> Optional[float]:
        """Score eating window using selected method and custom parameters."""
        if hours is None:
            return None
            
        params = st.session_state.formula_params
        
        if method == 'linear':
            p = params['window_linear']
            return max(0, min(100, (p['intercept'] - hours) * (-p['slope'])))
        elif method == 'tiered':
            p = params['window_tiered']
            for i, threshold in enumerate(p['thresholds']):
                if hours <= threshold:
                    return p['scores'][i]
            return p['scores'][-1]  # Min score if above all thresholds
        elif method == 'logarithmic':
            p = params['window_logistic']
            return round(p['max_score'] / (1 + math.exp(p['steepness'] * (hours - p['midpoint']))), 1)

    def score_eating_midpoint(self, midpoint_hour: Optional[float]) -> Optional[float]:
        """Score eating midpoint timing using custom optimal range."""
        if midpoint_hour is None:
            return None
            
        optimal = st.session_state.formula_params['midpoint_optimal']
        
        if optimal['start'] <= midpoint_hour <= optimal['end']:
            return 100
        elif optimal['start'] - 0.5 <= midpoint_hour <= optimal['end'] + 0.5:
            return 90
        elif optimal['start'] - 1.0 <= midpoint_hour <= optimal['end'] + 1.0:
            return 75
        elif optimal['start'] - 1.5 <= midpoint_hour <= optimal['end'] + 1.5:
            return 60
        elif optimal['start'] - 2.0 <= midpoint_hour <= optimal['end'] + 2.0:
            return 40
        else:
            return 20

    def calculate_consistency_penalty(self, midpoints: list) -> Tuple[Optional[float], Optional[int]]:
        """Calculate consistency modifier and penalty using custom thresholds."""
        if len(midpoints) < 4:
            return None, None
            
        # Filter out None values
        valid_midpoints = [mp for mp in midpoints if mp is not None]
        if len(valid_midpoints) < 4:
            return None, None
            
        std_dev = np.std(valid_midpoints)
        
        # Penalty based on custom thresholds
        thresholds = st.session_state.formula_params['consistency_thresholds']
        penalties = st.session_state.formula_params['consistency_penalties']
        
        for i, threshold in enumerate(thresholds):
            if std_dev <= threshold:
                return round(std_dev, 2), penalties[i]
                
        return round(std_dev, 2), penalties[-1]  # Max penalty

    def calculate_discipline_bonus(self, fasting_hours: Optional[float], 
                                 eating_window: Optional[float], 
                                 midpoint: Optional[float]) -> int:
        """Calculate discipline bonus using custom criteria."""
        if None in [fasting_hours, eating_window, midpoint]:
            return 0
            
        criteria = st.session_state.formula_params['discipline_criteria']
        
        # Check peak disciplined first
        peak = criteria['peak']
        if (fasting_hours >= peak['fasting'] and 
            eating_window <= peak['window'] and 
            peak['midpoint_start'] <= midpoint <= peak['midpoint_end']):
            return peak['bonus']
            
        # Check high disciplined
        high = criteria['high']
        if (fasting_hours >= high['fasting'] and 
            eating_window <= high['window'] and 
            high['midpoint_start'] <= midpoint <= high['midpoint_end']):
            return high['bonus']
            
        # Check good disciplined
        good = criteria['good']
        if (fasting_hours >= good['fasting'] and 
            eating_window <= good['window'] and 
            good['midpoint_start'] <= midpoint <= good['midpoint_end']):
            return good['bonus']
            
        return 0

    def process_data(self, scoring_method: str) -> pd.DataFrame:
        """Process all data and calculate scores."""
        df = st.session_state.entries.copy()
        if df.empty:
            return df
            
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate metrics
        df['fasting_duration'] = None
        df['eating_window'] = None
        df['eating_midpoint'] = None
        df['fasting_score'] = None
        df['window_score'] = None
        df['midpoint_score'] = None
        df['combined_score'] = None
        df['consistency_stdev'] = None
        df['consistency_penalty'] = None
        df['discipline_bonus'] = None
        df['final_score'] = None
        
        for i in range(len(df)):
            # Get previous day's last meal for fasting calculation
            prev_last_meal = df.iloc[i-1]['last_meal_time'] if i > 0 else None
            
            # Calculate basic metrics
            fasting_duration = self.calculate_fasting_duration(
                prev_last_meal, df.iloc[i]['first_meal_time'])
            eating_window = self.calculate_eating_window(
                df.iloc[i]['first_meal_time'], df.iloc[i]['last_meal_time'])
            eating_midpoint = self.calculate_eating_midpoint(
                df.iloc[i]['first_meal_time'], df.iloc[i]['last_meal_time'])
            
            df.at[i, 'fasting_duration'] = fasting_duration
            df.at[i, 'eating_window'] = eating_window
            df.at[i, 'eating_midpoint'] = eating_midpoint
            
            # Calculate scores
            df.at[i, 'fasting_score'] = self.score_fasting_duration(fasting_duration, scoring_method)
            df.at[i, 'window_score'] = self.score_eating_window(eating_window, scoring_method)
            df.at[i, 'midpoint_score'] = self.score_eating_midpoint(eating_midpoint)
            
            # Combined score (average of available scores)
            scores = [s for s in [df.at[i, 'fasting_score'], df.at[i, 'window_score'], 
                                df.at[i, 'midpoint_score']] if s is not None]
            df.at[i, 'combined_score'] = round(np.mean(scores), 1) if scores else None
            
            # Consistency calculation (last 4 days including current)
            if i >= 3:
                recent_midpoints = df.iloc[i-3:i+1]['eating_midpoint'].tolist()
                consistency_stdev, consistency_penalty = self.calculate_consistency_penalty(recent_midpoints)
                df.at[i, 'consistency_stdev'] = consistency_stdev
                df.at[i, 'consistency_penalty'] = consistency_penalty
            
            # Discipline bonus
            df.at[i, 'discipline_bonus'] = self.calculate_discipline_bonus(
                fasting_duration, eating_window, eating_midpoint)
            
            # Final score
            base_score = df.at[i, 'combined_score'] or 0
            penalty = df.at[i, 'consistency_penalty'] or 0
            bonus = df.at[i, 'discipline_bonus'] or 0
            df.at[i, 'final_score'] = round(base_score + penalty + bonus, 1) if base_score else None
            
        return df

    def render_formula_tuning(self):
        """Render the formula tuning interface."""
        st.header("🔧 Formula Tuning")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Fasting Duration", "Eating Window", "Midpoint Timing", 
            "Consistency", "Discipline Bonus"
        ])
        
        with tab1:
            st.subheader("Fasting Duration Scoring")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Linear Formula**")
                slope = st.number_input("Slope", value=st.session_state.formula_params['fasting_linear']['slope'], step=0.1, key="fast_linear_slope")
                intercept = st.number_input("Intercept (hours)", value=st.session_state.formula_params['fasting_linear']['intercept'], step=0.1, key="fast_linear_int")
                st.session_state.formula_params['fasting_linear'] = {'slope': slope, 'intercept': intercept}
                
            with col2:
                st.write("**Tiered Formula**")
                thresholds_str = st.text_area("Thresholds (comma-separated)", 
                    value=",".join(map(str, st.session_state.formula_params['fasting_tiered']['thresholds'])), key="fast_tier_thresh")
                scores_str = st.text_area("Scores (comma-separated)",
                    value=",".join(map(str, st.session_state.formula_params['fasting_tiered']['scores'])), key="fast_tier_scores")
                
                try:
                    thresholds = [float(x.strip()) for x in thresholds_str.split(',')]
                    scores = [float(x.strip()) for x in scores_str.split(',')]
                    if len(scores) == len(thresholds) + 1:
                        st.session_state.formula_params['fasting_tiered'] = {'thresholds': thresholds, 'scores': scores}
                    else:
                        st.error("Scores must have one more value than thresholds")
                except:
                    st.error("Invalid format for thresholds or scores")
                    
            with col3:
                st.write("**Logistic Formula**")
                steepness = st.number_input("Steepness", value=st.session_state.formula_params['fasting_logistic']['steepness'], step=0.1, key="fast_log_steep")
                midpoint = st.number_input("Midpoint (hours)", value=st.session_state.formula_params['fasting_logistic']['midpoint'], step=0.1, key="fast_log_mid")
                max_score = st.number_input("Max Score", value=st.session_state.formula_params['fasting_logistic']['max_score'], step=1, key="fast_log_max")
                st.session_state.formula_params['fasting_logistic'] = {'steepness': steepness, 'midpoint': midpoint, 'max_score': max_score}
        
        with tab2:
            st.subheader("Eating Window Scoring")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Linear Formula**")
                slope = st.number_input("Slope", value=st.session_state.formula_params['window_linear']['slope'], step=0.1, key="window_linear_slope")
                intercept = st.number_input("Intercept (hours)", value=st.session_state.formula_params['window_linear']['intercept'], step=0.1, key="window_linear_int")
                st.session_state.formula_params['window_linear'] = {'slope': slope, 'intercept': intercept}
                
            with col2:
                st.write("**Tiered Formula**")
                thresholds_str = st.text_area("Thresholds (comma-separated)", 
                    value=",".join(map(str, st.session_state.formula_params['window_tiered']['thresholds'])), key="window_tier_thresh")
                scores_str = st.text_area("Scores (comma-separated)",
                    value=",".join(map(str, st.session_state.formula_params['window_tiered']['scores'])), key="window_tier_scores")
                
                try:
                    thresholds = [float(x.strip()) for x in thresholds_str.split(',')]
                    scores = [float(x.strip()) for x in scores_str.split(',')]
                    if len(scores) == len(thresholds) + 1:
                        st.session_state.formula_params['window_tiered'] = {'thresholds': thresholds, 'scores': scores}
                    else:
                        st.error("Scores must have one more value than thresholds")
                except:
                    st.error("Invalid format for thresholds or scores")
                    
            with col3:
                st.write("**Logistic Formula**")
                steepness = st.number_input("Steepness", value=st.session_state.formula_params['window_logistic']['steepness'], step=0.1, key="window_log_steep")
                midpoint = st.number_input("Midpoint (hours)", value=st.session_state.formula_params['window_logistic']['midpoint'], step=0.1, key="window_log_mid")
                max_score = st.number_input("Max Score", value=st.session_state.formula_params['window_logistic']['max_score'], step=1, key="window_log_max")
                st.session_state.formula_params['window_logistic'] = {'steepness': steepness, 'midpoint': midpoint, 'max_score': max_score}
        
        with tab3:
            st.subheader("Optimal Eating Midpoint")
            
            col1, col2 = st.columns(2)
            with col1:
                start_hour = st.number_input("Optimal Start (24h format)", 
                    value=st.session_state.formula_params['midpoint_optimal']['start'], 
                    step=0.5, min_value=0.0, max_value=23.5, key="midpoint_start")
            with col2:
                end_hour = st.number_input("Optimal End (24h format)", 
                    value=st.session_state.formula_params['midpoint_optimal']['end'], 
                    step=0.5, min_value=0.0, max_value=23.5, key="midpoint_end")
                    
            st.session_state.formula_params['midpoint_optimal'] = {'start': start_hour, 'end': end_hour}
        
        with tab4:
            st.subheader("Consistency Thresholds & Penalties")
            
            col1, col2 = st.columns(2)
            with col1:
                thresholds_str = st.text_area("Standard Deviation Thresholds (hours)", 
                    value=",".join(map(str, st.session_state.formula_params['consistency_thresholds'])), key="consist_thresh")
            with col2:
                penalties_str = st.text_area("Penalties (negative values)", 
                    value=",".join(map(str, st.session_state.formula_params['consistency_penalties'])), key="consist_pen")
                    
            try:
                thresholds = [float(x.strip()) for x in thresholds_str.split(',')]
                penalties = [int(x.strip()) for x in penalties_str.split(',')]
                if len(penalties) == len(thresholds) + 1:
                    st.session_state.formula_params['consistency_thresholds'] = thresholds
                    st.session_state.formula_params['consistency_penalties'] = penalties
                else:
                    st.error("Penalties must have one more value than thresholds")
            except:
                st.error("Invalid format for thresholds or penalties")
        
        with tab5:
            st.subheader("Discipline Bonus Criteria")
            
            criteria = st.session_state.formula_params['discipline_criteria']
            
            for level in ['peak', 'high', 'good']:
                st.write(f"**{level.title()} Level**")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    fasting = st.number_input(f"Min Fasting (h)", value=criteria[level]['fasting'], step=1, key=f"disc_{level}_fast")
                with col2:
                    window = st.number_input(f"Max Window (h)", value=criteria[level]['window'], step=1, key=f"disc_{level}_window")
                with col3:
                    mid_start = st.number_input(f"Midpoint Start", value=criteria[level]['midpoint_start'], step=0.5, key=f"disc_{level}_start")
                with col4:
                    mid_end = st.number_input(f"Midpoint End", value=criteria[level]['midpoint_end'], step=0.5, key=f"disc_{level}_end")
                with col5:
                    bonus = st.number_input(f"Bonus Points", value=criteria[level]['bonus'], step=1, key=f"disc_{level}_bonus")
                
                st.session_state.formula_params['discipline_criteria'][level] = {
                    'fasting': fasting, 'window': window, 
                    'midpoint_start': mid_start, 'midpoint_end': mid_end, 'bonus': bonus
                }

    def render_behavior_tracker_config(self):
        """Render behavior tracker configuration."""
        st.header("🎯 Behavior Tracker Configuration")
        
        st.subheader("Current Trackers")
        
        trackers_to_remove = []
        for key, tracker in st.session_state.behavior_trackers.items():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
            
            with col1:
                new_name = st.text_input("Name", value=tracker['name'], key=f"tracker_name_{key}")
                tracker['name'] = new_name
                
            with col2:
                tracker_type = st.selectbox("Type", 
                    options=['binary', 'count', 'duration'], 
                    index=['binary', 'count', 'duration'].index(tracker['type']),
                    key=f"tracker_type_{key}")
                tracker['type'] = tracker_type
                
            with col3:
                if tracker_type == 'count':
                    max_val = st.number_input("Max", value=tracker.get('max_value', 5), min_value=1, key=f"tracker_max_{key}")
                    tracker['max_value'] = max_val
                elif tracker_type == 'duration':
                    unit = st.selectbox("Unit", options=['minutes', 'hours'], 
                        index=0 if tracker.get('unit', 'minutes') == 'minutes' else 1,
                        key=f"tracker_unit_{key}")
                    tracker['unit'] = unit
                else:
                    st.write("—")
                    
            with col4:
                tracker['active'] = st.checkbox("Active", value=tracker['active'], key=f"tracker_active_{key}")
                
            with col5:
                if st.button("🗑️", key=f"remove_tracker_{key}"):
                    trackers_to_remove.append(key)
        
        # Remove trackers
        for key in trackers_to_remove:
            del st.session_state.behavior_trackers[key]
            st.rerun()
        
        st.subheader("Add New Tracker")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_tracker_name = st.text_input("Tracker Name", key="new_tracker_name")
        with col2:
            new_tracker_type = st.selectbox("Type", options=['binary', 'count', 'duration'], key="new_tracker_type")
        with col3:
            if new_tracker_type == 'count':
                new_max_value = st.number_input("Max Value", value=5, min_value=1, key="new_tracker_max")
            elif new_tracker_type == 'duration':
                new_unit = st.selectbox("Unit", options=['minutes', 'hours'], key="new_tracker_unit")
        with col4:
            if st.button("➕ Add Tracker"):
                if new_tracker_name:
                    key = new_tracker_name.lower().replace(' ', '_')
                    new_tracker = {
                        'name': new_tracker_name,
                        'type': new_tracker_type,
                        'active': True
                    }
                    if new_tracker_type == 'count':
                        new_tracker['max_value'] = new_max_value
                    elif new_tracker_type == 'duration':
                        new_tracker['unit'] = new_unit
                    
                    st.session_state.behavior_trackers[key] = new_tracker
                    st.rerun()

    def render_subjective_metrics_config(self):
        """Render subjective metrics configuration."""
        st.header("😊 Subjective Metrics Configuration")
        
        st.subheader("Current Metrics")
        
        metrics_to_remove = []
        for key, metric in st.session_state.subjective_metrics.items():
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            
            with col1:
                new_name = st.text_input("Name", value=metric['name'], key=f"metric_name_{key}")
                metric['name'] = new_name
                
            with col2:
                min_val = st.number_input("Min", value=metric['scale'][0], key=f"metric_min_{key}")
                
            with col3:
                max_val = st.number_input("Max", value=metric['scale'][1], key=f"metric_max_{key}")
                metric['scale'] = [min_val, max_val]
                
            with col4:
                metric['active'] = st.checkbox("Active", value=metric['active'], key=f"metric_active_{key}")
                
            with col5:
                if st.button("🗑️", key=f"remove_metric_{key}"):
                    metrics_to_remove.append(key)
        
        # Remove metrics
        for key in metrics_to_remove:
            del st.session_state.subjective_metrics[key]
            st.rerun()
        
        st.subheader("Add New Metric")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_metric_name = st.text_input("Metric Name", key="new_metric_name")
        with col2:
            new_min = st.number_input("Min Value", value=1, key="new_metric_min")
        with col3:
            new_max = st.number_input("Max Value", value=10, key="new_metric_max")
        with col4:
            if st.button("➕ Add Metric"):
                if new_metric_name:
                    key = new_metric_name.lower().replace(' ', '_')
                    new_metric = {
                        'name': new_metric_name,
                        'scale': [new_min, new_max],
                        'active': True
                    }
                    st.session_state.subjective_metrics[key] = new_metric
                    st.rerun()

    def render_data_entry(self):
        """Render the data entry interface."""
        st.header("📝 Daily Entry")
        
        self.ensure_dataframe_columns()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Core Tracking")
            entry_date = st.date_input("Date", value=datetime.now().date())
            first_meal = st.time_input("First Meal Time", value=time(12, 0))
            last_meal = st.time_input("Last Meal Time", value=time(18, 0))
            sleep_time = st.time_input("Sleep Time", value=time(22, 0))
            
            st.subheader("Behavior Tracking")
            behavior_values = {}
            for key, tracker in st.session_state.behavior_trackers.items():
                if tracker['active']:
                    if tracker['type'] == 'binary':
                        behavior_values[key] = st.checkbox(tracker['name'], key=f"entry_{key}")
                    elif tracker['type'] == 'count':
                        behavior_values[key] = st.number_input(
                            tracker['name'], 
                            min_value=0, 
                            max_value=tracker.get('max_value', 10),
                            value=0, 
                            key=f"entry_{key}"
                        )
                    elif tracker['type'] == 'duration':
                        unit = tracker.get('unit', 'minutes')
                        behavior_values[key] = st.number_input(
                            f"{tracker['name']} ({unit})", 
                            min_value=0, 
                            value=0, 
                            key=f"entry_{key}"
                        )
            
        with col2:
            st.subheader("Subjective Metrics")
            subjective_values = {}
            for key, metric in st.session_state.subjective_metrics.items():
                if metric['active']:
                    min_val, max_val = metric['scale']
                    default_val = min_val + (max_val - min_val) // 2
                    subjective_values[key] = st.slider(
                        metric['name'], 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=default_val,
                        key=f"entry_{key}"
                    )
        
        if st.button("💾 Save Entry", type="primary"):
            new_entry = {
                'date': entry_date,
                'first_meal_time': first_meal,
                'last_meal_time': last_meal,
                'sleep_time': sleep_time,
                **behavior_values,
                **subjective_values
            }
            
            # Check if entry for this date already exists
            existing = st.session_state.entries[st.session_state.entries['date'] == entry_date]
            if not existing.empty:
                st.session_state.entries = st.session_state.entries[st.session_state.entries['date'] != entry_date]
            
            # Ensure all columns exist
            for col in self.get_active_columns():
                if col not in new_entry:
                    new_entry[col] = None
            
            st.session_state.entries = pd.concat([st.session_state.entries, pd.DataFrame([new_entry])], ignore_index=True)
            st.success(f"Entry saved for {entry_date}")
            st.rerun()

    def render_dashboard(self, scoring_method: str):
        """Render the dashboard interface."""
        st.header("📊 Dashboard")
        
        if st.session_state.entries.empty:
            st.info("No data yet. Add some entries in the Data Entry tab!")
            return
            
        processed_df = self.process_data(scoring_method)
        
        # Recent stats
        col1, col2, col3, col4 = st.columns(4)
        
        recent_data = processed_df.tail(7)  # Last 7 days
        
        with col1:
            avg_fasting = recent_data['fasting_duration'].mean()
            st.metric("Avg Fasting (7d)", f"{avg_fasting:.1f}h" if not pd.isna(avg_fasting) else "N/A")
            
        with col2:
            avg_window = recent_data['eating_window'].mean()
            st.metric("Avg Window (7d)", f"{avg_window:.1f}h" if not pd.isna(avg_window) else "N/A")
            
        with col3:
            avg_score = recent_data['final_score'].mean()
            st.metric("Avg Score (7d)", f"{avg_score:.1f}" if not pd.isna(avg_score) else "N/A")
            
        with col4:
            consistency = recent_data['consistency_stdev'].iloc[-1] if len(recent_data) > 0 else None
            st.metric("Consistency", f"{consistency:.2f}h" if consistency is not None else "N/A")
        
        # Behavior tracking summary
        if any(tracker['active'] for tracker in st.session_state.behavior_trackers.values()):
            st.subheader("Behavior Summary (Last 7 Days)")
            behavior_cols = []
            for key, tracker in st.session_state.behavior_trackers.items():
                if tracker['active'] and key in recent_data.columns:
                    behavior_cols.append(key)
            
            if behavior_cols:
                behavior_summary = recent_data[behavior_cols].sum()
                behavior_display = pd.DataFrame({
                    'Activity': [st.session_state.behavior_trackers[key]['name'] for key in behavior_cols],
                    'Total (7d)': [behavior_summary[key] for key in behavior_cols]
                })
                st.dataframe(behavior_display, use_container_width=True)
        
        # Subjective metrics summary
        if any(metric['active'] for metric in st.session_state.subjective_metrics.values()):
            st.subheader("Subjective Metrics (Last 7 Days)")
            subjective_cols = []
            for key, metric in st.session_state.subjective_metrics.items():
                if metric['active'] and key in recent_data.columns:
                    subjective_cols.append(key)
            
            if subjective_cols:
                subjective_summary = recent_data[subjective_cols].mean()
                subjective_display = pd.DataFrame({
                    'Metric': [st.session_state.subjective_metrics[key]['name'] for key in subjective_cols],
                    'Avg (7d)': [f"{subjective_summary[key]:.1f}" if not pd.isna(subjective_summary[key]) else "N/A" for key in subjective_cols]
                })
                st.dataframe(subjective_display, use_container_width=True)
        
        # Recent entries table
        st.subheader("Recent Entries")
        display_cols = ['date', 'fasting_duration', 'eating_window', 'eating_midpoint', 
                      'final_score', 'discipline_bonus', 'consistency_penalty']
        
        if not processed_df.empty:
            recent_display = processed_df[display_cols].tail(10).round(2)
            st.dataframe(recent_display, use_container_width=True)

    def render_analytics(self, scoring_method: str):
        """Render the analytics interface."""
        st.header("📈 Analytics")
        
        if st.session_state.entries.empty:
            st.info("No data available for analytics yet.")
            return
            
        processed_df = self.process_data(scoring_method)
        
        # Score trends
        st.subheader("Score Trends")
        fig_scores = go.Figure()
        fig_scores.add_trace(go.Scatter(
            x=processed_df['date'], 
            y=processed_df['final_score'],
            mode='lines+markers',
            name='Final Score',
            line=dict(color='blue')
        ))
        fig_scores.add_trace(go.Scatter(
            x=processed_df['date'], 
            y=processed_df['combined_score'],
            mode='lines+markers',
            name='Base Score',
            line=dict(color='green')
        ))
        fig_scores.update_layout(title="Score Trends", yaxis_title="Score")
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Fasting patterns
        col1, col2 = st.columns(2)
        
        with col1:
            fig_fasting = px.line(processed_df, x='date', y='fasting_duration', 
                                title='Fasting Duration Trend')
            st.plotly_chart(fig_fasting, use_container_width=True)
            
        with col2:
            fig_window = px.line(processed_df, x='date', y='eating_window',
                               title='Eating Window Trend')
            st.plotly_chart(fig_window, use_container_width=True)
        
        # Behavior tracking charts
        active_behaviors = [key for key, tracker in st.session_state.behavior_trackers.items() 
                          if tracker['active'] and key in processed_df.columns]
        
        if active_behaviors:
            st.subheader("Behavior Trends")
            behavior_cols = min(2, len(active_behaviors))
            behavior_chart_cols = st.columns(behavior_cols)
            
            for i, behavior_key in enumerate(active_behaviors[:4]):  # Show max 4 charts
                tracker = st.session_state.behavior_trackers[behavior_key]
                col_idx = i % behavior_cols
                
                with behavior_chart_cols[col_idx]:
                    if tracker['type'] == 'binary':
                        # Bar chart for binary data
                        fig = px.bar(processed_df, x='date', y=behavior_key, 
                                   title=f"{tracker['name']} Activity")
                    else:
                        # Line chart for count/duration data
                        fig = px.line(processed_df, x='date', y=behavior_key, 
                                    title=f"{tracker['name']} Trend")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Subjective metrics charts
        active_subjective = [key for key, metric in st.session_state.subjective_metrics.items() 
                           if metric['active'] and key in processed_df.columns]
        
        if active_subjective:
            st.subheader("Subjective Metrics Trends")
            subjective_cols = min(2, len(active_subjective))
            subjective_chart_cols = st.columns(subjective_cols)
            
            for i, metric_key in enumerate(active_subjective[:4]):  # Show max 4 charts
                metric = st.session_state.subjective_metrics[metric_key]
                col_idx = i % subjective_cols
                
                with subjective_chart_cols[col_idx]:
                    fig = px.line(processed_df, x='date', y=metric_key, 
                                title=f"{metric['name']} Trend")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        if len(processed_df) > 10:  # Only show if enough data
            st.subheader("Correlation Analysis")
            
            # Select numeric columns for correlation
            numeric_cols = ['fasting_duration', 'eating_window', 'final_score']
            numeric_cols.extend(active_behaviors)
            numeric_cols.extend(active_subjective)
            
            correlation_df = processed_df[numeric_cols].corr()
            
            fig_corr = px.imshow(correlation_df, 
                               title="Correlation Matrix",
                               color_continuous_scale="RdBu_r",
                               aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)

def main():
    st.set_page_config(page_title="IF Tracker Pro", page_icon="⏱️", layout="wide")
    
    tracker = IFTracker()
    
    st.title("⏱️ Intermittent Fasting Tracker Pro")
    st.markdown("*Advanced tracking with customizable formulas and behavior tracking*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        scoring_method = st.selectbox(
            "Scoring System",
            options=list(tracker.scoring_systems.keys()),
            format_func=lambda x: tracker.scoring_systems[x],
            index=1  # Default to tiered
        )
        
        st.header("Quick Actions")
        if st.button("📖 Scoring Explanations"):
            st.session_state.show_explanations = not st.session_state.get('show_explanations', False)
        
        if st.button("🔄 Reset All Data"):
            if st.button("⚠️ Confirm Reset"):
                st.session_state.entries = pd.DataFrame()
                st.success("All data cleared!")
                st.rerun()
        
        # Formula preview
        with st.expander("Current Formula Settings"):
            st.write(f"**Fasting Linear:** slope={st.session_state.formula_params['fasting_linear']['slope']}")
            st.write(f"**Window Optimal:** {st.session_state.formula_params['midpoint_optimal']['start']}-{st.session_state.formula_params['midpoint_optimal']['end']}h")
            st.write(f"**Active Behaviors:** {sum(1 for t in st.session_state.behavior_trackers.values() if t['active'])}")
            st.write(f"**Active Metrics:** {sum(1 for m in st.session_state.subjective_metrics.values() if m['active'])}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Data Entry", "📊 Dashboard", "📈 Analytics", 
        "🔧 Formula Tuning", "🎯 Behaviors", "😊 Metrics"
    ])
    
    with tab1:
        tracker.render_data_entry()
    
    with tab2:
        tracker.render_dashboard(scoring_method)
    
    with tab3:
        tracker.render_analytics(scoring_method)
        
    with tab4:
        tracker.render_formula_tuning()
        
    with tab5:
        tracker.render_behavior_tracker_config()
        
    with tab6:
        tracker.render_subjective_metrics_config()
    
    # Show explanations if requested
    if st.session_state.get('show_explanations', False):
        with st.expander("📖 Scoring System Explanations", expanded=True):
            for metric, info in tracker.explanations.items():
                st.subheader(metric.replace('_', ' ').title())
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Optimal Range:** {info['optimal_range']}")
                st.write(f"**Benefits:** {info['benefits']}")
                st.write("---")
    
    # Export functionality
    if not st.session_state.entries.empty:
        st.sidebar.header("Export")
        processed_df = tracker.process_data(scoring_method)
        
        # Format for export
        export_df = processed_df.copy()
        export_df['date'] = export_df['date'].astype(str)
        
        # Convert time columns to strings
        time_cols = ['first_meal_time', 'last_meal_time', 'sleep_time']
        for col in time_cols:
            if col in export_df.columns:
                export_df[col] = export_df[col].astype(str)
        
        csv = export_df.to_csv(index=False)
        
        st.sidebar.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"if_tracker_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Export settings
        settings_json = json.dumps({
            'formula_params': st.session_state.formula_params,
            'behavior_trackers': st.session_state.behavior_trackers,
            'subjective_metrics': st.session_state.subjective_metrics
        }, indent=2)
        
        st.sidebar.download_button(
            label="⚙️ Download Settings",
            data=settings_json,
            file_name=f"if_tracker_settings_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
