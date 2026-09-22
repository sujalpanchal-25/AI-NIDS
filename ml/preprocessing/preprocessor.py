"""
AI-NIDS ML Preprocessing Pipeline
Data preprocessing and feature engineering for network intrusion detection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Optional, Union, Any
import pickle
import os
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for network traffic data.
    Handles CICIDS2017 and UNSW-NB15 datasets.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.fitted = False
        
        # Default feature columns (CICIDS2017 compatible)
        self.default_features = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
            'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
            'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
            'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
            'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
            'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
            'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
            'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
            'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
            'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
            'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
            'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std',
            'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
        
    def load_dataset(self, file_path: str, dataset_type: str = 'cicids') -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to the CSV file
            dataset_type: Type of dataset ('cicids' or 'unsw')
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading dataset from {file_path}")
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and infinite values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        return df
    
    def identify_columns(self, df: pd.DataFrame) -> None:
        """
        Identify numerical and categorical columns.
        
        Args:
            df: Input DataFrame
        """
        self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove label column from features if present
        label_candidates = ['Label', 'label', 'attack_cat', 'Attack', 'class']
        for col in label_candidates:
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
        
        logger.info(f"Identified {len(self.numerical_columns)} numerical, {len(self.categorical_columns)} categorical columns")
    
    def extract_features(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract specified features from the dataset.
        
        Args:
            df: Input DataFrame
            features: List of feature column names (uses defaults if None)
            
        Returns:
            DataFrame with selected features
        """
        if features is None:
            features = self.default_features
        
        # Find available features
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features[:5]}...")
        
        self.feature_columns = available_features
        return df[available_features].copy()
    
    def extract_labels(self, df: pd.DataFrame, label_column: str = 'Label') -> np.ndarray:
        """
        Extract and encode labels.
        
        Args:
            df: Input DataFrame
            label_column: Name of the label column
            
        Returns:
            Encoded labels array
        """
        if label_column not in df.columns:
            # Try alternative names
            alternatives = ['label', 'attack_cat', 'Attack', 'class']
            for alt in alternatives:
                if alt in df.columns:
                    label_column = alt
                    break
            else:
                raise ValueError(f"Label column not found. Available: {df.columns.tolist()}")
        
        labels = df[label_column].values
        
        # Encode labels
        if not self.fitted:
            self.label_encoder.fit(labels)
        
        encoded_labels = self.label_encoder.transform(labels)
        
        # Log label distribution
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"Label distribution: {dict(zip(unique, counts))}")
        
        return encoded_labels
    
    def create_binary_labels(self, df: pd.DataFrame, label_column: str = 'Label') -> np.ndarray:
        """
        Create binary labels (normal vs attack).
        
        Args:
            df: Input DataFrame
            label_column: Name of the label column
            
        Returns:
            Binary labels (0: normal, 1: attack)
        """
        if label_column not in df.columns:
            alternatives = ['label', 'attack_cat', 'Attack', 'class']
            for alt in alternatives:
                if alt in df.columns:
                    label_column = alt
                    break
        
        labels = df[label_column].str.lower()
        binary_labels = (~labels.isin(['benign', 'normal', 'legitimate'])).astype(int)
        
        logger.info(f"Binary labels: {sum(binary_labels == 0)} normal, {sum(binary_labels == 1)} attack")
        return binary_labels.values
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            X: Input features
            
        Returns:
            Scaled features
        """
        logger.info("Fitting and transforming data...")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Handle missing and infinite values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Impute remaining missing values
        X = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.fitted = True
        logger.info(f"Transformed data shape: {X_scaled.shape}")
        
        return X_scaled
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Input features
            
        Returns:
            Scaled features
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Handle missing and infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Impute if fitted
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.imputer)
            X = self.imputer.transform(X)
        except Exception:
            pass
        
        # Scale if fitted
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self.scaler)
            X = self.scaler.transform(X)
        except Exception:
            pass

        return X
    
    def extract_features_from_live_flow(self, flow: Dict[str, Any]) -> np.ndarray:
        """
        Convert a live network flow dictionary into the exact ML feature vector
        matching the expected feature columns and trained scaling transformation.
        
        Args:
            flow: Dictionary containing live flow metrics (e.g. from FlowTracker.to_dict())
            
        Returns:
            2D numpy array of shape (1, num_features) with preprocessed/scaled features
        """
        # Determine target feature columns
        feature_cols = self.feature_columns
        if not feature_cols:
            # Attempt to load feature columns from models directory if available
            import os, pickle
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            feat_path = os.path.join(base_dir, 'models', 'feature_columns.pkl')
            if os.path.exists(feat_path):
                try:
                    with open(feat_path, 'rb') as f:
                        feature_cols = pickle.load(f)
                    self.feature_columns = feature_cols
                except Exception:
                    feature_cols = self.default_features
            else:
                feature_cols = self.default_features

        # Extract core fields from live flow
        duration = float(flow.get('duration', flow.get('dur', flow.get('flow_duration', 0.0))))
        duration_safe = max(duration, 0.0001)

        packets_sent = float(flow.get('packets_sent', flow.get('packets_forward', flow.get('spkts', flow.get('count', 0.0)))))
        packets_recv = float(flow.get('packets_recv', flow.get('packets_backward', flow.get('dpkts', flow.get('srv_count', 0.0)))))
        total_packets = float(flow.get('total_packets', packets_sent + packets_recv))
        if total_packets == 0.0 and (packets_sent > 0 or packets_recv > 0):
            total_packets = packets_sent + packets_recv

        bytes_sent = float(flow.get('bytes_sent', flow.get('sbytes', flow.get('src_bytes', 0.0))))
        bytes_recv = float(flow.get('bytes_recv', flow.get('dbytes', flow.get('dst_bytes', 0.0))))

        syn_count = float(flow.get('syn_count', 0.0))
        rst_count = float(flow.get('rst_count', 0.0))

        src_ip = str(flow.get('source_ip', flow.get('src_ip', '')))
        dst_ip = str(flow.get('destination_ip', flow.get('dst_ip', '')))
        src_port = float(flow.get('source_port', flow.get('src_port', 0)))
        dst_port = float(flow.get('destination_port', flow.get('dst_port', 0)))

        iat_mean = float(flow.get('iat_mean', 0.0))
        iat_std = float(flow.get('iat_std', 0.0))

        # Calculate derived statistics matching ML feature definitions
        rate = total_packets / duration_safe
        sload = (bytes_sent * 8.0) / duration_safe
        dload = (bytes_recv * 8.0) / duration_safe

        sinpkt = (duration * 1000.0) / max(packets_sent, 1.0) if packets_sent > 0 else (iat_mean * 1000.0)
        dinpkt = (duration * 1000.0) / max(packets_recv, 1.0) if packets_recv > 0 else 0.0

        smean = bytes_sent / max(packets_sent, 1.0) if packets_sent > 0 else float(flow.get('packet_length_mean', 0.0))
        dmean = bytes_recv / max(packets_recv, 1.0) if packets_recv > 0 else 0.0

        serror_rate = syn_count / max(packets_sent, 1.0) if syn_count > 0 else float(flow.get('serror_rate', 0.0))
        rerror_rate = rst_count / max(total_packets, 1.0) if rst_count > 0 else float(flow.get('rerror_rate', 0.0))

        is_sm_ips_ports = 1.0 if (src_ip and dst_ip and src_ip == dst_ip and src_port == dst_port and src_port != 0) else 0.0
        land = is_sm_ips_ports
        is_ftp_login = 1.0 if (src_port == 21 or dst_port == 21) else 0.0
        ct_flw_http_mthd = 1.0 if dst_port in [80, 443, 8080] else 0.0

        # Build feature map for standard ML feature names
        feature_map = {
            'duration': duration,
            'packets_sent': packets_sent,
            'packets_recv': packets_recv,
            'bytes_sent': bytes_sent,
            'bytes_recv': bytes_recv,
            'rate': rate,
            'src_ttl': float(flow.get('src_ttl', 64.0)),
            'dst_ttl': float(flow.get('dst_ttl', 64.0)),
            'sload': sload,
            'dload': dload,
            'sloss': float(flow.get('sloss', rst_count)),
            'dloss': float(flow.get('dloss', 0.0)),
            'sinpkt': sinpkt,
            'dinpkt': dinpkt,
            'sjit': float(flow.get('sjit', iat_std * 1000.0)),
            'djit': float(flow.get('djit', 0.0)),
            'swin': float(flow.get('swin', 255.0)),
            'stcpb': float(flow.get('stcpb', 0.0)),
            'dtcpb': float(flow.get('dtcpb', 0.0)),
            'dwin': float(flow.get('dwin', 255.0)),
            'tcprtt': float(flow.get('tcprtt', 0.0)),
            'synack': float(flow.get('synack', 0.0)),
            'ackdat': float(flow.get('ackdat', 0.0)),
            'smean': smean,
            'dmean': dmean,
            'trans_depth': float(flow.get('trans_depth', 0.0)),
            'response_body_len': float(flow.get('response_body_len', 0.0)),
            'ct_srv_src': float(flow.get('ct_srv_src', 1.0)),
            'ct_state_ttl': float(flow.get('ct_state_ttl', 0.0)),
            'ct_dst_ltm': float(flow.get('ct_dst_ltm', 1.0)),
            'ct_src_dport_ltm': float(flow.get('ct_src_dport_ltm', dst_port)),
            'ct_dst_sport_ltm': float(flow.get('ct_dst_sport_ltm', src_port)),
            'ct_dst_src_ltm': float(flow.get('ct_dst_src_ltm', 1.0)),
            'is_ftp_login': float(flow.get('is_ftp_login', is_ftp_login)),
            'ct_ftp_cmd': float(flow.get('ct_ftp_cmd', 0.0)),
            'ct_flw_http_mthd': float(flow.get('ct_flw_http_mthd', ct_flw_http_mthd)),
            'ct_src_ltm': float(flow.get('ct_src_ltm', 1.0)),
            'ct_srv_dst': float(flow.get('ct_srv_dst', 1.0)),
            'is_sm_ips_ports': float(flow.get('is_sm_ips_ports', is_sm_ips_ports)),
            'land': float(flow.get('land', land)),
            'wrong_fragment': float(flow.get('wrong_fragment', 0.0)),
            'urgent': float(flow.get('urgent', flow.get('urg_count', 0.0))),
            'hot': float(flow.get('hot', 0.0)),
            'num_failed_logins': float(flow.get('num_failed_logins', 0.0)),
            'logged_in': float(flow.get('logged_in', 0.0)),
            'num_compromised': float(flow.get('num_compromised', 0.0)),
            'root_shell': float(flow.get('root_shell', 0.0)),
            'su_attempted': float(flow.get('su_attempted', 0.0)),
            'num_root': float(flow.get('num_root', 0.0)),
            'num_file_creations': float(flow.get('num_file_creations', 0.0)),
            'num_shells': float(flow.get('num_shells', 0.0)),
            'num_access_files': float(flow.get('num_access_files', 0.0)),
            'num_outbound_cmds': float(flow.get('num_outbound_cmds', 0.0)),
            'is_host_login': float(flow.get('is_host_login', 0.0)),
            'is_guest_login': float(flow.get('is_guest_login', 0.0)),
            'serror_rate': serror_rate,
            'srv_serror_rate': float(flow.get('srv_serror_rate', 0.0)),
            'rerror_rate': rerror_rate,
            'srv_rerror_rate': float(flow.get('srv_rerror_rate', 0.0)),
            'same_srv_rate': float(flow.get('same_srv_rate', 1.0)),
            'diff_srv_rate': float(flow.get('diff_srv_rate', 0.0)),
            'srv_diff_host_rate': float(flow.get('srv_diff_host_rate', 0.0)),
            'dst_host_count': float(flow.get('dst_host_count', 1.0)),
            'dst_host_srv_count': float(flow.get('dst_host_srv_count', 1.0)),
            'dst_host_same_srv_rate': float(flow.get('dst_host_same_srv_rate', 1.0)),
            'dst_host_diff_srv_rate': float(flow.get('dst_host_diff_srv_rate', 0.0)),
            'dst_host_same_src_port_rate': float(flow.get('dst_host_same_src_port_rate', 0.0)),
            'dst_host_srv_diff_host_rate': float(flow.get('dst_host_srv_diff_host_rate', 0.0)),
            'dst_host_serror_rate': float(flow.get('dst_host_serror_rate', 0.0)),
            'dst_host_srv_serror_rate': float(flow.get('dst_host_srv_serror_rate', 0.0)),
            'dst_host_rerror_rate': float(flow.get('dst_host_rerror_rate', 0.0)),
            'dst_host_srv_rerror_rate': float(flow.get('dst_host_srv_rerror_rate', 0.0)),
        }

        # Map features into exact column order
        raw_values = []
        for col in feature_cols:
            if col in feature_map:
                raw_values.append(feature_map[col])
            elif col in flow:
                try:
                    raw_values.append(float(flow[col]))
                except (ValueError, TypeError):
                    raw_values.append(0.0)
            else:
                raw_values.append(0.0)

        X_raw = np.array([raw_values], dtype=np.float32)

        # Apply fitted scaling transformation if preprocessor/scaler is fitted
        return self.transform(X_raw)

    def prepare_data(
        self,
        df: pd.DataFrame,
        label_column: str = 'Label',
        binary: bool = True,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline.
        
        Args:
            df: Input DataFrame
            label_column: Name of the label column
            binary: Whether to use binary classification
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Preparing data pipeline...")
        
        # Clean data
        df = self.clean_data(df)
        
        # Identify columns
        self.identify_columns(df)
        
        # Extract features
        X = self.extract_features(df)
        
        # Extract labels
        if binary:
            y = self.create_binary_labels(df, label_column)
        else:
            y = self.extract_labels(df, label_column)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Further split for validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        # Scale features
        X_train = self.fit_transform(X_train)
        X_val = self.transform(X_val)
        X_test = self.transform(X_test)
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_sequence_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for LSTM model.
        
        Args:
            X: Input features
            y: Labels
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - sequence_length + 1):
            X_sequences.append(X[i:i + sequence_length])
            y_sequences.append(y[i + sequence_length - 1])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def save(self, path: str) -> None:
        """Save preprocessor state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'fitted': self.fitted,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved preprocessor to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """Load preprocessor from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls()
        if isinstance(state, list):
            preprocessor.feature_columns = state
            preprocessor.numerical_columns = state
            preprocessor.fitted = True
            logger.info(f"Loaded feature columns preprocessor from list at {path}")
            return preprocessor

        if isinstance(state, dict):
            preprocessor.config = state.get('config')
            preprocessor.scaler = state.get('scaler')
            preprocessor.label_encoder = state.get('label_encoder')
            preprocessor.imputer = state.get('imputer')
            preprocessor.feature_columns = state.get('feature_columns', [])
            preprocessor.numerical_columns = state.get('numerical_columns', [])
            preprocessor.categorical_columns = state.get('categorical_columns', [])
            preprocessor.fitted = state.get('fitted', True)
        
        logger.info(f"Loaded preprocessor from {path}")
        return preprocessor


class FeatureEngineer:
    """
    Feature engineering for network traffic data.
    Creates derived features for improved detection.
    """
    
    @staticmethod
    def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        
        # Packet ratio features
        if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
            total = df['Total Fwd Packets'] + df['Total Backward Packets']
            df['Fwd_Packet_Ratio'] = df['Total Fwd Packets'] / (total + 1e-7)
            df['Bwd_Packet_Ratio'] = df['Total Backward Packets'] / (total + 1e-7)
        
        # Byte features
        if 'Total Length of Fwd Packets' in df.columns and 'Total Length of Bwd Packets' in df.columns:
            total_bytes = df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']
            df['Bytes_Per_Packet'] = total_bytes / (df.get('Total Fwd Packets', 1) + df.get('Total Backward Packets', 1) + 1e-7)
        
        return df
    
    @staticmethod
    def add_time_features(df: pd.DataFrame, timestamp_col: str = 'Timestamp') -> pd.DataFrame:
        """Add time-based features."""
        
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            df['Hour'] = df[timestamp_col].dt.hour
            df['DayOfWeek'] = df[timestamp_col].dt.dayofweek
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
            df['IsBusinessHours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
        
        return df
    
    @staticmethod
    def add_flag_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add flag-based features."""
        
        flag_cols = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 
                     'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count']
        
        available_flags = [c for c in flag_cols if c in df.columns]
        
        if available_flags:
            df['Total_Flags'] = df[available_flags].sum(axis=1)
            df['Flag_Diversity'] = (df[available_flags] > 0).sum(axis=1)
        
        return df


def create_preprocessor(config: Optional[Dict] = None) -> DataPreprocessor:
    """Factory function to create a DataPreprocessor instance."""
    return DataPreprocessor(config)
