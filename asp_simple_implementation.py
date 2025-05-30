import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
import os
warnings.filterwarnings('ignore')

class ASPKeystrokeAuthenticator:
    """
    Adaptive Statistical Profile (ASP) for Keystroke Authentication
    Implementation for the paper: "Session-Adaptive Keystroke Dynamics: 
    Outlier-Resilient Profiling with Robust Statistics"
    
    Author: Yuji Umemoto, Kwassui Women's University
    """
    
    def __init__(self, data_path: str = 'DSL-StrongPasswordData.csv', random_seed: int = 42):
        """
        Initialize ASP authenticator
        
        Args:
            data_path: Path to the CMU Keystroke Dynamics Dataset
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset file '{data_path}' not found.\n"
                f"Please download the CMU Keystroke Dynamics Dataset from:\n"
                f"https://www.cs.cmu.edu/~keystroke/\n"
                f"Save as '{data_path}' in the current directory."
            )
        
        print(f"Loading dataset: {data_path}")
        self.df = pd.read_csv(data_path)
        
        # CMU dataset 31 timing features (H.*, DD.*, UD.*)
        self.features = [col for col in self.df.columns 
                        if col.startswith(('H.', 'DD.', 'UD.'))]
        self.features.sort()  # Consistent ordering
        
        print(f"Features: {len(self.features)} timing features")
        print(f"Dataset: {len(self.df['subject'].unique())} users, {len(self.df)} samples")
    
    def create_robust_profile(self, samples: pd.DataFrame) -> Dict:
        """
        Create robust user profile using adaptive statistical measures
        
        Args:
            samples: Training samples for a specific user
            
        Returns:
            Dictionary containing robust statistical profile
        """
        template = {}
        
        for feature in self.features:
            values = samples[feature].dropna().values
            if len(values) < 2:
                template[feature] = None
                continue
                
            # Robust statistics (outlier-resistant)
            median = np.median(values)
            q1, q3 = np.percentile(values, [25, 75])
            iqr = max(q3 - q1, 1e-10)  # Interquartile Range
            mad = np.median(np.abs(values - median))  # Median Absolute Deviation
            mad = max(mad, 1e-10)
            
            # Classical statistics (for comparison)
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            std = max(std, 1e-10)
            
            # Adaptive weights based on stability and reliability
            cv = std / (abs(mean) + 1e-10)  # Coefficient of Variation
            stability_weight = 1.0 / (1.0 + cv)  # Higher weight for stable features
            reliability_weight = min(1.0, np.sqrt(len(values)) / 10.0)  # Higher weight for more samples
            composite_weight = stability_weight * reliability_weight
            
            template[feature] = {
                'median': median,
                'iqr': iqr,
                'mad': mad,
                'mean': mean,
                'std': std,
                'stability_weight': stability_weight,
                'reliability_weight': reliability_weight,
                'composite_weight': composite_weight,
                'n_samples': len(values)
            }
            
        return template
    
    def asp_distance(self, template: Dict, sample: pd.Series, weight_type: str = 'composite') -> float:
        """
        Calculate ASP distance using multi-metric robust similarity
        
        Args:
            template: User profile template
            sample: Test sample
            weight_type: Type of weighting ('composite' or 'uniform')
            
        Returns:
            ASP distance score
        """
        if template is None:
            return float('inf')
            
        weighted_distance = 0.0
        total_weight = 0.0
        valid_features = 0
        
        for feature in self.features:
            if (template.get(feature) is None or 
                pd.isna(sample[feature])):
                continue
                
            feat_template = template[feature]
            value = sample[feature]
            
            # Weight selection
            if weight_type == 'composite':
                weight = feat_template['composite_weight']
            else:
                weight = 1.0
            
            # Three robust distance components
            median_dist = abs(value - feat_template['median']) / feat_template['iqr']
            mad_dist = abs(value - feat_template['median']) / feat_template['mad']
            std_dist = abs(value - feat_template['mean']) / feat_template['std']
            
            # Composite distance with weighted combination (0.5, 0.3, 0.2)
            # Emphasizes median-based robustness while maintaining compatibility
            composite_dist = 0.5 * median_dist + 0.3 * mad_dist + 0.2 * std_dist
            
            weighted_distance += composite_dist * weight
            total_weight += weight
            valid_features += 1
        
        if total_weight <= 0 or valid_features == 0:
            return float('inf')
            
        return weighted_distance / total_weight
    
    def evaluate_user_authentication(self, user_subject: str, training_ratio: float = 0.5) -> Dict:
        """
        Evaluate authentication performance for a single user
        
        Args:
            user_subject: Target user ID
            training_ratio: Ratio of data used for training (0.5 = 50%)
            
        Returns:
            Dictionary containing EER, threshold, FRR, and FAR
        """
        # Get target user data
        user_data = self.df[self.df['subject'] == user_subject].copy()
        user_data = user_data.sort_values(['sessionIndex', 'rep']).reset_index(drop=True)
        
        if len(user_data) < 20:
            return None
        
        # Train-test split maintaining temporal order
        n_train = int(len(user_data) * training_ratio)
        n_train = max(10, min(n_train, len(user_data) - 10))
        
        train_data = user_data.iloc[:n_train]
        test_data = user_data.iloc[n_train:]
        
        # Create ASP profile
        asp_template = self.create_robust_profile(train_data)
        
        # Calculate genuine scores (legitimate user)
        genuine_scores = []
        for _, sample in test_data.iterrows():
            score = self.asp_distance(asp_template, sample, 'composite')
            genuine_scores.append(score)
        
        # Calculate impostor scores (other users)
        other_subjects = [s for s in self.df['subject'].unique() if s != user_subject]
        impostor_scores = []
        
        # Balanced impostor sampling
        max_impostors_per_user = max(1, len(test_data) // len(other_subjects))
        
        for impostor_subject in other_subjects:
            impostor_data = self.df[self.df['subject'] == impostor_subject]
            
            if len(impostor_data) == 0:
                continue
            
            # Deterministic sampling for reproducibility
            n_samples = min(max_impostors_per_user, len(impostor_data))
            np.random.seed(hash(user_subject + impostor_subject) % 2**32)
            sample_indices = np.random.choice(len(impostor_data), n_samples, replace=False)
            
            for idx in sample_indices:
                sample = impostor_data.iloc[idx]
                score = self.asp_distance(asp_template, sample, 'composite')
                impostor_scores.append(score)
        
        # Calculate Equal Error Rate (EER)
        eer_result = self._calculate_eer(genuine_scores, impostor_scores)
        
        return {
            'eer': eer_result['eer'],
            'threshold': eer_result['threshold'],
            'frr': eer_result['frr'],
            'far': eer_result['far']
        }
    
    def _calculate_eer(self, genuine_scores: List[float], impostor_scores: List[float]) -> Dict:
        """
        Calculate Equal Error Rate (EER) and optimal threshold
        
        Args:
            genuine_scores: Distance scores for legitimate user
            impostor_scores: Distance scores for impostors
            
        Returns:
            Dictionary with EER, threshold, FRR, and FAR
        """
        if not genuine_scores or not impostor_scores:
            return {'eer': 1.0, 'threshold': float('inf'), 'frr': 1.0, 'far': 1.0}
        
        # Remove infinite values
        genuine_finite = [s for s in genuine_scores if np.isfinite(s)]
        impostor_finite = [s for s in impostor_scores if np.isfinite(s)]
        
        if not genuine_finite or not impostor_finite:
            return {'eer': 1.0, 'threshold': float('inf'), 'frr': 1.0, 'far': 1.0}
        
        all_scores = sorted(set(genuine_finite + impostor_finite))
        
        min_eer = 1.0
        best_result = {'eer': 1.0, 'threshold': float('inf'), 'frr': 1.0, 'far': 1.0}
        
        for threshold in all_scores:
            # Distance-based: accept if distance <= threshold
            frr = sum(1 for s in genuine_finite if s > threshold) / len(genuine_finite)
            far = sum(1 for s in impostor_finite if s <= threshold) / len(impostor_finite)
            eer = (frr + far) / 2.0
            
            if eer < min_eer:
                min_eer = eer
                best_result = {
                    'eer': eer,
                    'threshold': threshold,
                    'frr': frr,
                    'far': far
                }
        
        return best_result
    
    def run_full_evaluation(self, training_ratio: float = 0.5) -> Dict:
        """
        Run complete evaluation across all users
        
        Args:
            training_ratio: Ratio of data used for training
            
        Returns:
            Dictionary containing aggregated results
        """
        print("="*80)
        print("ASP KEYSTROKE AUTHENTICATION EVALUATION")
        print("Implementation for Academic Paper")
        print("="*80)
        
        subjects = sorted(self.df['subject'].unique())
        print(f"\nEvaluating {len(subjects)} users")
        print(f"Training ratio: {training_ratio:.1%}")
        
        all_eer_results = []
        successful_evaluations = 0
        
        # Process each user
        for user_idx, user_subject in enumerate(subjects):
            print(f"\n[{user_idx+1}/{len(subjects)}] User: {user_subject}")
            
            user_result = self.evaluate_user_authentication(user_subject, training_ratio)
            
            if user_result is None:
                print("Evaluation failed: insufficient data")
                continue
            
            all_eer_results.append(user_result['eer'])
            successful_evaluations += 1
            
            print(f"ASP EER: {user_result['eer']*100:.2f}%")
        
        print(f"\nSuccessful evaluations: {successful_evaluations}/{len(subjects)}")
        
        # Calculate final statistics
        if all_eer_results:
            mean_eer = np.mean(all_eer_results)
            std_eer = np.std(all_eer_results, ddof=1)
            median_eer = np.median(all_eer_results)
            
            print(f"\n=== FINAL RESULTS ===")
            print(f"Mean EER: {mean_eer*100:.2f}% ± {std_eer*100:.2f}%")
            print(f"Median EER: {median_eer*100:.2f}%")
            
            return {
                'mean_eer': mean_eer,
                'std_eer': std_eer,
                'median_eer': median_eer,
                'n_users': successful_evaluations,
                'all_eers': all_eer_results
            }
        else:
            return {'mean_eer': 1.0}
    
    def evaluate_performance(self, data_path: str = None) -> Dict:
        """
        Main evaluation interface for external use
        
        Args:
            data_path: Optional path to different dataset
            
        Returns:
            Dictionary containing performance metrics
        """
        if data_path and data_path != 'DSL-StrongPasswordData.csv':
            self.__init__(data_path, 42)
        
        # Run evaluation with paper parameters
        results = self.run_full_evaluation(training_ratio=0.5)
        
        # Return results in standard format
        if 'mean_eer' in results and results['mean_eer'] < 1.0:
            return {
                'eer': results['mean_eer'],
                'frr': results['mean_eer'],  # Approximate FRR at EER point
                'far': results['mean_eer'],  # Approximate FAR at EER point
                'threshold': 0.0,  # Placeholder
                'std_eer': results.get('std_eer', 0.0),
                'median_eer': results.get('median_eer', results['mean_eer']),
                'n_users': results.get('n_users', 0)
            }
        else:
            return {'eer': 1.0, 'frr': 1.0, 'far': 1.0}


def main():
    """
    Main execution function
    Reproduces the results reported in the academic paper
    """
    try:
        print("="*60)
        print("ASP Keystroke Authentication System")
        print("Author: Yuji Umemoto, Kwassui Women's University")
        print("="*60)
        
        # Initialize with CMU dataset
        asp = ASPKeystrokeAuthenticator('DSL-StrongPasswordData.csv', random_seed=42)
        
        print("\nRunning comprehensive evaluation...")
        results = asp.evaluate_performance()
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Equal Error Rate (EER): {results['eer']*100:.2f}%")
        if 'std_eer' in results:
            print(f"Standard Deviation: ±{results['std_eer']*100:.2f}%")
        if 'median_eer' in results:
            print(f"Median EER: {results['median_eer']*100:.2f}%")
        if 'n_users' in results:
            print(f"Users evaluated: {results['n_users']}")
        print("="*60)
        
        return results
        
    except FileNotFoundError as e:
        print("ERROR: Dataset not found")
        print(str(e))
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


if __name__ == "__main__":
    results = main()