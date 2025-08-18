import numpy as np
import pandas as pd
import pickle
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple
import json

def load_and_separate_data(pickle_path: str) -> Dict:
    """
    Load the pickle data and separate it into different aspects
    """
    with open(pickle_path, 'rb') as f:
        dat = pickle.load(f)
    
    # Get unique values for categorization
    unique_days = sorted(set(dat['post_implant_day']))
    unique_vocab_sizes = sorted(set(dat['vocab_size']))
    
    # Create trial-level metadata
    trials_data = []
    phoneme_instances = []
    performance_metrics = []
    
    for trial_idx in range(len(dat['cue_sentence'])):
        # Trial metadata
        trial_data = {
            'trial_id': trial_idx,
            'post_implant_day': dat['post_implant_day'][trial_idx],
            'vocab_size': dat['vocab_size'][trial_idx],
            'cue_sentence': dat['cue_sentence'][trial_idx],
            'decoded_sentence': dat['decoded_sentence'][trial_idx],
            'trial_length': len(dat['decoded_logits'][trial_idx])
        }
        trials_data.append(trial_data)
        
        # Phoneme-level data
        cue_phonemes = dat['cue_sentence_phonemes'][trial_idx]
        decoded_phonemes = dat['decoded_phonemes_raw'][trial_idx]
        logits = dat['decoded_logits'][trial_idx]
        
        # Create phoneme instances with timing
        for time_step, (cue_phone, decoded_phone, logit_vector) in enumerate(
            zip(cue_phonemes, decoded_phonemes, logits)
        ):
            phoneme_instance = {
                'trial_id': trial_idx,
                'time_step': time_step,
                'cue_phoneme': cue_phone,
                'decoded_phoneme': decoded_phone,
                'is_correct': cue_phone == decoded_phone,
                'logit_values': json.dumps(logit_vector.tolist()),
                'max_logit_value': float(np.max(logit_vector)),
                'min_logit_value': float(np.min(logit_vector)),
                'logit_std': float(np.std(logit_vector)),
                'confidence_score': float(np.exp(np.max(logit_vector)) / np.sum(np.exp(logit_vector)))
            }
            phoneme_instances.append(phoneme_instance)
        
        # Performance metrics per trial
        total_phonemes = len(cue_phonemes)
        correct_phonemes = sum(1 for c, d in zip(cue_phonemes, decoded_phonemes) if c == d)
        phoneme_error_rate = (total_phonemes - correct_phonemes) / total_phonemes if total_phonemes > 0 else 1.0
        
        # Word-level metrics
        cue_words = dat['cue_sentence'][trial_idx].split()
        decoded_words = dat['decoded_sentence'][trial_idx].split()
        total_words = len(cue_words)
        correct_words = sum(1 for c, d in zip(cue_words, decoded_words) if c == d)
        word_error_rate = (total_words - correct_words) / total_words if total_words > 0 else 1.0
        
        performance_metric = {
            'trial_id': trial_idx,
            'total_phonemes': total_phonemes,
            'correct_phonemes': correct_phonemes,
            'phoneme_error_rate': phoneme_error_rate,
            'total_words': total_words,
            'correct_words': correct_words,
            'word_error_rate': word_error_rate,
            'trial_duration': len(logits)
        }
        performance_metrics.append(performance_metric)
    
    return {
        'trials': trials_data,
        'phoneme_instances': phoneme_instances,
        'performance_metrics': performance_metrics,
        'unique_days': unique_days,
        'unique_vocab_sizes': unique_vocab_sizes
    }

def create_database_schema(db_path: str):
    """
    Create SQLite database with relational schema for phoneme analysis
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript('''
        -- Trials table (main experimental sessions)
        CREATE TABLE IF NOT EXISTS trials (
            trial_id INTEGER PRIMARY KEY,
            post_implant_day INTEGER,
            vocab_size INTEGER,
            cue_sentence TEXT,
            decoded_sentence TEXT,
            trial_length INTEGER,
            session_date TEXT,
            corpus_type TEXT,
            split_type TEXT
        );
        
        -- Phoneme instances table (individual phoneme predictions)
        CREATE TABLE IF NOT EXISTS phoneme_instances (
            instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id INTEGER,
            time_step INTEGER,
            cue_phoneme TEXT,
            decoded_phoneme TEXT,
            is_correct BOOLEAN,
            logit_values TEXT,  -- JSON string of all logit values
            max_logit_value REAL,
            min_logit_value REAL,
            logit_std REAL,
            confidence_score REAL,
            FOREIGN KEY (trial_id) REFERENCES trials (trial_id)
        );
        
        -- Performance metrics table
        CREATE TABLE IF NOT EXISTS performance_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id INTEGER,
            total_phonemes INTEGER,
            correct_phonemes INTEGER,
            phoneme_error_rate REAL,
            total_words INTEGER,
            correct_words INTEGER,
            word_error_rate REAL,
            trial_duration INTEGER,
            FOREIGN KEY (trial_id) REFERENCES trials (trial_id)
        );
        
        -- Phoneme characteristics table
        CREATE TABLE IF NOT EXISTS phoneme_characteristics (
            phoneme_id INTEGER PRIMARY KEY AUTOINCREMENT,
            phoneme TEXT UNIQUE,
            phoneme_type TEXT,  -- vowel, consonant, silence, blank
            is_vowel BOOLEAN,
            is_consonant BOOLEAN,
            is_silence BOOLEAN,
            logit_index INTEGER
        );
        
        -- Time-based analysis table
        CREATE TABLE IF NOT EXISTS temporal_analysis (
            temporal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id INTEGER,
            time_window_start INTEGER,
            time_window_end INTEGER,
            window_size INTEGER,
            avg_confidence REAL,
            avg_error_rate REAL,
            phoneme_diversity INTEGER,
            FOREIGN KEY (trial_id) REFERENCES trials (trial_id)
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_trials_day ON trials(post_implant_day);
        CREATE INDEX IF NOT EXISTS idx_trials_vocab ON trials(vocab_size);
        CREATE INDEX IF NOT EXISTS idx_phoneme_trial ON phoneme_instances(trial_id);
        CREATE INDEX IF NOT EXISTS idx_phoneme_correct ON phoneme_instances(is_correct);
        CREATE INDEX IF NOT EXISTS idx_phoneme_cue ON phoneme_instances(cue_phoneme);
        CREATE INDEX IF NOT EXISTS idx_phoneme_decoded ON phoneme_instances(decoded_phoneme);
    ''')
    
    conn.commit()
    conn.close()

def populate_phoneme_characteristics(db_path: str):
    """
    Populate the phoneme characteristics table
    """
    # Define phoneme characteristics
    phoneme_defs = [
        ('BLANK', 'blank', False, False, False, 0),
        ('SIL', 'silence', False, False, True, 1),
        ('AA', 'vowel', True, False, False, 2),
        ('AE', 'vowel', True, False, False, 3),
        ('AH', 'vowel', True, False, False, 4),
        ('AO', 'vowel', True, False, False, 5),
        ('AW', 'vowel', True, False, False, 6),
        ('AY', 'vowel', True, False, False, 7),
        ('B', 'consonant', False, True, False, 8),
        ('CH', 'consonant', False, True, False, 9),
        ('D', 'consonant', False, True, False, 10),
        ('DH', 'consonant', False, True, False, 11),
        ('EH', 'vowel', True, False, False, 12),
        ('ER', 'vowel', True, False, False, 13),
        ('EY', 'vowel', True, False, False, 14),
        ('F', 'consonant', False, True, False, 15),
        ('G', 'consonant', False, True, False, 16),
        ('HH', 'consonant', False, True, False, 17),
        ('IH', 'vowel', True, False, False, 18),
        ('IY', 'vowel', True, False, False, 19),
        ('JH', 'consonant', False, True, False, 20),
        ('K', 'consonant', False, True, False, 21),
        ('L', 'consonant', False, True, False, 22),
        ('M', 'consonant', False, True, False, 23),
        ('N', 'consonant', False, True, False, 24),
        ('NG', 'consonant', False, True, False, 25),
        ('OW', 'vowel', True, False, False, 26),
        ('OY', 'vowel', True, False, False, 27),
        ('P', 'consonant', False, True, False, 28),
        ('R', 'consonant', False, True, False, 29),
        ('S', 'consonant', False, True, False, 30),
        ('SH', 'consonant', False, True, False, 31),
        ('T', 'consonant', False, True, False, 32),
        ('TH', 'consonant', False, True, False, 33),
        ('UH', 'vowel', True, False, False, 34),
        ('UW', 'vowel', True, False, False, 35),
        ('V', 'consonant', False, True, False, 36),
        ('W', 'consonant', False, True, False, 37),
        ('Y', 'consonant', False, True, False, 38),
        ('Z', 'consonant', False, True, False, 39),
        ('ZH', 'consonant', False, True, False, 40)
    ]
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.executemany('''
        INSERT OR REPLACE INTO phoneme_characteristics 
        (phoneme, phoneme_type, is_vowel, is_consonant, is_silence, logit_index)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', phoneme_defs)
    
    conn.commit()
    conn.close()

def populate_database(separated_data: Dict, db_path: str, csv_data_path: str = None):
    """
    Populate the database with the separated data
    """
    # Load CSV data for additional metadata
    if csv_data_path:
        csv_data = pd.read_csv(csv_data_path)
    else:
        csv_data = None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Insert trials data
    for trial in separated_data['trials']:
        # Add additional metadata from CSV if available
        session_date = None
        corpus_type = None
        split_type = None
        
        if csv_data is not None:
            # Create mask for matching entries
            day_mask = csv_data['Post-implant day'] == trial['post_implant_day']
            corpus_name = f"{trial['vocab_size']}-Word" if trial['vocab_size'] == 50 else 'Switchboard'
            corpus_mask = csv_data['Corpus'] == corpus_name
            combined_mask = day_mask & corpus_mask
            csv_entry = csv_data[combined_mask]
            if not csv_entry.empty:
                session_date = csv_entry.iloc[0]['Date']
                corpus_type = csv_entry.iloc[0]['Corpus']
                split_type = csv_entry.iloc[0]['Split']
        
        cursor.execute('''
            INSERT INTO trials 
            (trial_id, post_implant_day, vocab_size, cue_sentence, decoded_sentence, 
             trial_length, session_date, corpus_type, split_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trial['trial_id'], trial['post_implant_day'], trial['vocab_size'],
            trial['cue_sentence'], trial['decoded_sentence'], trial['trial_length'],
            session_date, corpus_type, split_type
        ))
    
    # Insert phoneme instances data
    for instance in separated_data['phoneme_instances']:
        cursor.execute('''
            INSERT INTO phoneme_instances 
            (trial_id, time_step, cue_phoneme, decoded_phoneme, is_correct,
             logit_values, max_logit_value, min_logit_value, logit_std, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            instance['trial_id'], instance['time_step'], instance['cue_phoneme'],
            instance['decoded_phoneme'], instance['is_correct'], instance['logit_values'],
            instance['max_logit_value'], instance['min_logit_value'], 
            instance['logit_std'], instance['confidence_score']
        ))
    
    # Insert performance metrics data
    for metric in separated_data['performance_metrics']:
        cursor.execute('''
            INSERT INTO performance_metrics 
            (trial_id, total_phonemes, correct_phonemes, phoneme_error_rate,
             total_words, correct_words, word_error_rate, trial_duration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric['trial_id'], metric['total_phonemes'], metric['correct_phonemes'],
            metric['phoneme_error_rate'], metric['total_words'], metric['correct_words'],
            metric['word_error_rate'], metric['trial_duration']
        ))
    
    conn.commit()
    conn.close()

def create_kumo_analysis_views(db_path: str):
    """
    Create useful views for Kumo AI analysis
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # View 1: Phoneme accuracy by day and vocabulary size
    cursor.execute('''
        CREATE VIEW IF NOT EXISTS phoneme_accuracy_by_conditions AS
        SELECT 
            t.post_implant_day,
            t.vocab_size,
            pc.phoneme_type,
            p.cue_phoneme,
            COUNT(*) as total_instances,
            SUM(CASE WHEN p.is_correct THEN 1 ELSE 0 END) as correct_instances,
            AVG(p.confidence_score) as avg_confidence,
            AVG(p.logit_std) as avg_logit_std
        FROM phoneme_instances p
        JOIN trials t ON p.trial_id = t.trial_id
        JOIN phoneme_characteristics pc ON p.cue_phoneme = pc.phoneme
        GROUP BY t.post_implant_day, t.vocab_size, pc.phoneme_type, p.cue_phoneme
    ''')
    
    # View 2: Temporal analysis of confidence and accuracy
    cursor.execute('''
        CREATE VIEW IF NOT EXISTS temporal_confidence_analysis AS
        SELECT 
            t.trial_id,
            t.post_implant_day,
            t.vocab_size,
            p.time_step,
            AVG(p.confidence_score) OVER (
                PARTITION BY t.trial_id 
                ORDER BY p.time_step 
                ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
            ) as moving_avg_confidence,
            AVG(CASE WHEN p.is_correct THEN 1.0 ELSE 0.0 END) OVER (
                PARTITION BY t.trial_id 
                ORDER BY p.time_step 
                ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
            ) as moving_avg_accuracy
        FROM phoneme_instances p
        JOIN trials t ON p.trial_id = t.trial_id
    ''')
    
    # View 3: Error patterns and confusion matrix
    cursor.execute('''
        CREATE VIEW IF NOT EXISTS phoneme_confusion_matrix AS
        SELECT 
            p.cue_phoneme as true_phoneme,
            p.decoded_phoneme as predicted_phoneme,
            COUNT(*) as frequency,
            pc1.phoneme_type as true_type,
            pc2.phoneme_type as predicted_type
        FROM phoneme_instances p
        JOIN phoneme_characteristics pc1 ON p.cue_phoneme = pc1.phoneme
        JOIN phoneme_characteristics pc2 ON p.decoded_phoneme = pc2.phoneme
        WHERE p.cue_phoneme != p.decoded_phoneme
        GROUP BY p.cue_phoneme, p.decoded_phoneme
        ORDER BY frequency DESC
    ''')
    
    # View 4: Performance trends over time
    cursor.execute('''
        CREATE VIEW IF NOT EXISTS performance_trends AS
        SELECT 
            t.post_implant_day,
            t.vocab_size,
            COUNT(DISTINCT t.trial_id) as num_trials,
            AVG(pm.phoneme_error_rate) as avg_phoneme_error_rate,
            AVG(pm.word_error_rate) as avg_word_error_rate,
            AVG(p.avg_confidence) as avg_confidence,
            AVG(p.avg_logit_std) as avg_logit_std
        FROM trials t
        JOIN performance_metrics pm ON t.trial_id = pm.trial_id
        JOIN (
            SELECT 
                trial_id,
                AVG(confidence_score) as avg_confidence,
                AVG(logit_std) as avg_logit_std
            FROM phoneme_instances
            GROUP BY trial_id
        ) p ON t.trial_id = p.trial_id
        GROUP BY t.post_implant_day, t.vocab_size
        ORDER BY t.post_implant_day, t.vocab_size
    ''')
    
    conn.commit()
    conn.close()

def main():
    """
    Main function to create the complete database
    """
    # File paths
    pickle_path = 'data/t15_copyTask.pkl'
    csv_path = 'data/t15_copyTaskData_description.csv'
    db_path = 'data/phoneme_analysis.db'
    
    print("Loading and separating data...")
    separated_data = load_and_separate_data(pickle_path)
    
    print("Creating database schema...")
    create_database_schema(db_path)
    
    print("Populating phoneme characteristics...")
    populate_phoneme_characteristics(db_path)
    
    print("Populating database with data...")
    populate_database(separated_data, db_path, csv_path)
    
    print("Creating analysis views...")
    create_kumo_analysis_views(db_path)
    
    print(f"Database created successfully at {db_path}")
    print(f"Total trials: {len(separated_data['trials'])}")
    print(f"Total phoneme instances: {len(separated_data['phoneme_instances'])}")
    print(f"Unique post-implant days: {separated_data['unique_days']}")
    print(f"Vocabulary sizes: {separated_data['unique_vocab_sizes']}")

if __name__ == "__main__":
    main()
