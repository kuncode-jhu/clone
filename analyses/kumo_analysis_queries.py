import sqlite3
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class PhonemeDatabaseAnalyzer:
    """
    Class to analyze the phoneme database for Kumo AI
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def get_database_summary(self) -> Dict:
        """
        Get a comprehensive summary of the database
        """
        cursor = self.conn.cursor()
        
        # Get table counts
        tables = ['trials', 'phoneme_instances', 'performance_metrics', 'phoneme_characteristics']
        counts = {}
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        
        # Get unique values
        cursor.execute("SELECT COUNT(DISTINCT post_implant_day) FROM trials")
        unique_days = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT vocab_size) FROM trials")
        unique_vocab_sizes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT cue_phoneme) FROM phoneme_instances")
        unique_phonemes = cursor.fetchone()[0]
        
        return {
            'table_counts': counts,
            'unique_days': unique_days,
            'unique_vocab_sizes': unique_vocab_sizes,
            'unique_phonemes': unique_phonemes
        }
    
    def get_phoneme_accuracy_by_type(self) -> pd.DataFrame:
        """
        Get phoneme accuracy broken down by phoneme type (vowel, consonant, etc.)
        """
        query = """
        SELECT 
            pc.phoneme_type,
            pc.phoneme,
            COUNT(*) as total_instances,
            SUM(CASE WHEN p.is_correct THEN 1 ELSE 0 END) as correct_instances,
            ROUND(100.0 * SUM(CASE WHEN p.is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_percent,
            AVG(p.confidence_score) as avg_confidence,
            AVG(p.logit_std) as avg_logit_std
        FROM phoneme_instances p
        JOIN phoneme_characteristics pc ON p.cue_phoneme = pc.phoneme
        GROUP BY pc.phoneme_type, pc.phoneme
        ORDER BY pc.phoneme_type, accuracy_percent DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_temporal_analysis(self, trial_id: int = None) -> pd.DataFrame:
        """
        Get temporal analysis of confidence and accuracy over time
        """
        if trial_id:
            query = """
            SELECT 
                t.trial_id,
                t.post_implant_day,
                t.vocab_size,
                p.time_step,
                p.confidence_score,
                p.is_correct,
                p.cue_phoneme,
                p.decoded_phoneme
            FROM phoneme_instances p
            JOIN trials t ON p.trial_id = t.trial_id
            WHERE t.trial_id = ?
            ORDER BY p.time_step
            """
            return pd.read_sql_query(query, self.conn, params=[trial_id])
        else:
            query = """
            SELECT 
                t.post_implant_day,
                t.vocab_size,
                AVG(p.confidence_score) as avg_confidence,
                AVG(CASE WHEN p.is_correct THEN 1.0 ELSE 0.0 END) as avg_accuracy,
                COUNT(*) as total_phonemes
            FROM phoneme_instances p
            JOIN trials t ON p.trial_id = t.trial_id
            GROUP BY t.post_implant_day, t.vocab_size
            ORDER BY t.post_implant_day, t.vocab_size
            """
            return pd.read_sql_query(query, self.conn)
    
    def get_confusion_matrix(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get the most common phoneme confusion patterns
        """
        query = """
        SELECT 
            p.cue_phoneme as true_phoneme,
            p.decoded_phoneme as predicted_phoneme,
            pc1.phoneme_type as true_type,
            pc2.phoneme_type as predicted_type,
            COUNT(*) as frequency,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM phoneme_instances p
        JOIN phoneme_characteristics pc1 ON p.cue_phoneme = pc1.phoneme
        JOIN phoneme_characteristics pc2 ON p.decoded_phoneme = pc2.phoneme
        WHERE p.cue_phoneme != p.decoded_phoneme
        GROUP BY p.cue_phoneme, p.decoded_phoneme
        ORDER BY frequency DESC
        LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=[top_n])
    
    def get_performance_by_conditions(self) -> pd.DataFrame:
        """
        Get performance metrics broken down by experimental conditions
        """
        query = """
        SELECT 
            t.post_implant_day,
            t.vocab_size,
            t.corpus_type,
            t.split_type,
            COUNT(DISTINCT t.trial_id) as num_trials,
            AVG(pm.phoneme_error_rate) as avg_phoneme_error_rate,
            AVG(pm.word_error_rate) as avg_word_error_rate,
            AVG(pm.total_phonemes) as avg_phonemes_per_trial,
            AVG(pm.total_words) as avg_words_per_trial
        FROM trials t
        JOIN performance_metrics pm ON t.trial_id = pm.trial_id
        GROUP BY t.post_implant_day, t.vocab_size, t.corpus_type, t.split_type
        ORDER BY t.post_implant_day, t.vocab_size
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_confidence_analysis(self) -> pd.DataFrame:
        """
        Analyze confidence scores and their relationship to accuracy
        """
        query = """
        SELECT 
            CASE 
                WHEN p.confidence_score < 0.3 THEN 'Low (<0.3)'
                WHEN p.confidence_score < 0.7 THEN 'Medium (0.3-0.7)'
                ELSE 'High (>0.7)'
            END as confidence_level,
            COUNT(*) as total_instances,
            SUM(CASE WHEN p.is_correct THEN 1 ELSE 0 END) as correct_instances,
            ROUND(100.0 * SUM(CASE WHEN p.is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_percent,
            AVG(p.confidence_score) as avg_confidence,
            AVG(p.logit_std) as avg_logit_std
        FROM phoneme_instances p
        GROUP BY confidence_level
        ORDER BY avg_confidence
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_phoneme_difficulty_ranking(self) -> pd.DataFrame:
        """
        Rank phonemes by difficulty (accuracy)
        """
        query = """
        SELECT 
            pc.phoneme,
            pc.phoneme_type,
            COUNT(*) as total_instances,
            SUM(CASE WHEN p.is_correct THEN 1 ELSE 0 END) as correct_instances,
            ROUND(100.0 * SUM(CASE WHEN p.is_correct THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_percent,
            AVG(p.confidence_score) as avg_confidence,
            AVG(p.logit_std) as avg_logit_std
        FROM phoneme_instances p
        JOIN phoneme_characteristics pc ON p.cue_phoneme = pc.phoneme
        WHERE pc.phoneme NOT IN ('BLANK', 'SIL')
        GROUP BY pc.phoneme, pc.phoneme_type
        HAVING total_instances >= 10
        ORDER BY accuracy_percent ASC
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_trial_level_analysis(self, limit: int = 100) -> pd.DataFrame:
        """
        Get trial-level analysis with performance metrics
        """
        query = """
        SELECT 
            t.trial_id,
            t.post_implant_day,
            t.vocab_size,
            t.corpus_type,
            t.split_type,
            t.cue_sentence,
            t.decoded_sentence,
            pm.phoneme_error_rate,
            pm.word_error_rate,
            pm.total_phonemes,
            pm.total_words,
            AVG(p.confidence_score) as avg_confidence,
            AVG(p.logit_std) as avg_logit_std
        FROM trials t
        JOIN performance_metrics pm ON t.trial_id = pm.trial_id
        JOIN phoneme_instances p ON t.trial_id = p.trial_id
        GROUP BY t.trial_id
        ORDER BY pm.phoneme_error_rate DESC
        LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=[limit])
    
    def get_logit_analysis(self, phoneme: str = None) -> pd.DataFrame:
        """
        Analyze logit distributions for specific phonemes or overall
        """
        if phoneme:
            query = """
            SELECT 
                p.cue_phoneme,
                p.decoded_phoneme,
                p.is_correct,
                p.max_logit_value,
                p.min_logit_value,
                p.logit_std,
                p.confidence_score,
                t.post_implant_day,
                t.vocab_size
            FROM phoneme_instances p
            JOIN trials t ON p.trial_id = t.trial_id
            WHERE p.cue_phoneme = ?
            ORDER BY p.confidence_score DESC
            """
            return pd.read_sql_query(query, self.conn, params=[phoneme])
        else:
            query = """
            SELECT 
                pc.phoneme_type,
                AVG(p.max_logit_value) as avg_max_logit,
                AVG(p.min_logit_value) as avg_min_logit,
                AVG(p.logit_std) as avg_logit_std,
                AVG(p.confidence_score) as avg_confidence,
                COUNT(*) as total_instances
            FROM phoneme_instances p
            JOIN phoneme_characteristics pc ON p.cue_phoneme = pc.phoneme
            GROUP BY pc.phoneme_type
            ORDER BY avg_confidence DESC
            """
            return pd.read_sql_query(query, self.conn)
    
    def export_for_kumo(self, output_path: str = "phoneme_analysis_for_kumo.csv"):
        """
        Export a comprehensive dataset for Kumo AI analysis
        """
        query = """
        SELECT 
            t.trial_id,
            t.post_implant_day,
            t.vocab_size,
            t.corpus_type,
            t.split_type,
            p.time_step,
            p.cue_phoneme,
            p.decoded_phoneme,
            p.is_correct,
            p.confidence_score,
            p.max_logit_value,
            p.min_logit_value,
            p.logit_std,
            pc.phoneme_type,
            pc.is_vowel,
            pc.is_consonant,
            pc.is_silence,
            pm.phoneme_error_rate as trial_phoneme_error_rate,
            pm.word_error_rate as trial_word_error_rate
        FROM phoneme_instances p
        JOIN trials t ON p.trial_id = t.trial_id
        JOIN phoneme_characteristics pc ON p.cue_phoneme = pc.phoneme
        JOIN performance_metrics pm ON t.trial_id = pm.trial_id
        ORDER BY t.trial_id, p.time_step
        """
        
        df = pd.read_sql_query(query, self.conn)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} records to {output_path}")
        return df
    
    def close(self):
        """
        Close the database connection
        """
        self.conn.close()

def create_kumo_analysis_report(analyzer: PhonemeDatabaseAnalyzer, output_file: str = "kumo_analysis_report.txt"):
    """
    Create a comprehensive analysis report for Kumo AI
    """
    with open(output_file, 'w') as f:
        f.write("PHONEME DATABASE ANALYSIS REPORT FOR KUMO AI\n")
        f.write("=" * 50 + "\n\n")
        
        # Database summary
        summary = analyzer.get_database_summary()
        f.write("DATABASE SUMMARY:\n")
        f.write(f"Total trials: {summary['table_counts']['trials']}\n")
        f.write(f"Total phoneme instances: {summary['table_counts']['phoneme_instances']}\n")
        f.write(f"Unique post-implant days: {summary['unique_days']}\n")
        f.write(f"Unique vocabulary sizes: {summary['unique_vocab_sizes']}\n")
        f.write(f"Unique phonemes: {summary['unique_phonemes']}\n\n")
        
        # Phoneme accuracy by type
        f.write("PHONEME ACCURACY BY TYPE:\n")
        accuracy_df = analyzer.get_phoneme_accuracy_by_type()
        f.write(accuracy_df.to_string(index=False))
        f.write("\n\n")
        
        # Most difficult phonemes
        f.write("MOST DIFFICULT PHONEMES (Lowest Accuracy):\n")
        difficulty_df = analyzer.get_phoneme_difficulty_ranking()
        f.write(difficulty_df.head(10).to_string(index=False))
        f.write("\n\n")
        
        # Confidence analysis
        f.write("CONFIDENCE ANALYSIS:\n")
        confidence_df = analyzer.get_confidence_analysis()
        f.write(confidence_df.to_string(index=False))
        f.write("\n\n")
        
        # Top confusion patterns
        f.write("TOP PHONEME CONFUSION PATTERNS:\n")
        confusion_df = analyzer.get_confusion_matrix(top_n=15)
        f.write(confusion_df.to_string(index=False))
        f.write("\n\n")
        
        # Performance by conditions
        f.write("PERFORMANCE BY EXPERIMENTAL CONDITIONS:\n")
        performance_df = analyzer.get_performance_by_conditions()
        f.write(performance_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("END OF REPORT\n")
    
    print(f"Analysis report saved to {output_file}")

def main():
    """
    Main function to demonstrate the analysis capabilities
    """
    db_path = 'data/phoneme_analysis.db'
    
    try:
        analyzer = PhonemeDatabaseAnalyzer(db_path)
        
        # Get database summary
        summary = analyzer.get_database_summary()
        print("Database Summary:")
        print(f"Total trials: {summary['table_counts']['trials']}")
        print(f"Total phoneme instances: {summary['table_counts']['phoneme_instances']}")
        print(f"Unique phonemes: {summary['unique_phonemes']}")
        print()
        
        # Export data for Kumo AI
        print("Exporting data for Kumo AI...")
        analyzer.export_for_kumo("phoneme_analysis_for_kumo.csv")
        
        # Create analysis report
        print("Creating analysis report...")
        create_kumo_analysis_report(analyzer)
        
        # Example queries
        print("\nExample Analysis Results:")
        print("1. Phoneme accuracy by type:")
        accuracy_df = analyzer.get_phoneme_accuracy_by_type()
        print(accuracy_df.head())
        
        print("\n2. Most difficult phonemes:")
        difficulty_df = analyzer.get_phoneme_difficulty_ranking()
        print(difficulty_df.head())
        
        print("\n3. Confidence analysis:")
        confidence_df = analyzer.get_confidence_analysis()
        print(confidence_df)
        
        analyzer.close()
        
    except FileNotFoundError:
        print(f"Database file not found at {db_path}")
        print("Please run create_phoneme_database.py first to create the database.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
