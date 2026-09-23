"""
AI-NIDS Configuration Module
============================
Central configuration for the entire application.
Supports multiple environments: development, testing, production.
"""

import os
from datetime import timedelta
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Ensure matplotlib uses a writable directory inside the project
mpl_dir = BASE_DIR / 'data' / '.matplotlib'
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(mpl_dir))


class Config:
    """Base configuration class."""
    
    # Application
    APP_NAME = "AI-NIDS"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "AI-Powered Network Intrusion Detection System"
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ai-nids-super-secret-key-change-in-production'
    
    # Database
    _db_url = os.environ.get('DATABASE_URL')
    if _db_url and _db_url.startswith('sqlite:///'):
        _sqlite_path = _db_url[10:]
        if not os.path.isabs(_sqlite_path) and not (len(_sqlite_path) > 1 and _sqlite_path[1] == ':'):
            SQLALCHEMY_DATABASE_URI = f'sqlite:///{BASE_DIR / _sqlite_path}'
        else:
            SQLALCHEMY_DATABASE_URI = _db_url
    elif _db_url:
        SQLALCHEMY_DATABASE_URI = _db_url
    else:
        SQLALCHEMY_DATABASE_URI = f'sqlite:///{BASE_DIR / "data" / "nids.db"}'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    
    # Session & Remember Me Cookie Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=2)
    REMEMBER_COOKIE_DURATION = timedelta(days=2)
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_REFRESH_EACH_REQUEST = True
    
    # Security
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = BASE_DIR / 'data' / 'logs' / 'nids.log'
    
    # ML Models
    MODELS_DIR = BASE_DIR / 'data' / 'saved_models'
    XGBOOST_MODEL_PATH = MODELS_DIR / 'xgboost_classifier.pkl'
    AUTOENCODER_MODEL_PATH = MODELS_DIR / 'autoencoder.pt'
    LSTM_MODEL_PATH = MODELS_DIR / 'lstm_detector.pt'
    SCALER_PATH = MODELS_DIR / 'scaler.pkl'
    LABEL_ENCODER_PATH = MODELS_DIR / 'label_encoder.pkl'
    
    # Detection Settings
    ANOMALY_THRESHOLD = 0.7
    ENSEMBLE_WEIGHTS = {
        'suricata': 0.4,
        'autoencoder': 0.3,
        'xgboost': 0.3
    }
    
    # Severity Thresholds
    SEVERITY_THRESHOLDS = {
        'critical': 0.9,
        'high': 0.7,
        'medium': 0.5,
        'low': 0.3,
        'info': 0.0
    }
    
    # Alert Settings
    MAX_ALERTS_DISPLAY = 100
    ALERT_RETENTION_DAYS = 30
    
    # Data Paths
    DATA_DIR = BASE_DIR / 'data'
    DATASETS_DIR = DATA_DIR / 'datasets'
    PROCESSED_DIR = DATA_DIR / 'processed'
    LOGS_DIR = DATA_DIR / 'logs'
    
    # Feature Columns (CICIDS2017 compatible)
    FEATURE_COLUMNS = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate'
    ]
    
    # CICIDS2017 Feature Columns
    CICIDS_FEATURE_COLUMNS = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
        'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
        'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
        'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
        'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
        'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
        'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
        'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
        'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
        'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
        'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
        'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
        'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
        'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std',
        'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
    ]
    
    # Attack Categories
    ATTACK_CATEGORIES = {
        'BENIGN': 'Normal',
        'DoS Hulk': 'DoS',
        'DoS GoldenEye': 'DoS',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'DDoS': 'DDoS',
        'PortScan': 'Probe',
        'FTP-Patator': 'Brute Force',
        'SSH-Patator': 'Brute Force',
        'Bot': 'Botnet',
        'Web Attack – Brute Force': 'Web Attack',
        'Web Attack – XSS': 'Web Attack',
        'Web Attack – Sql Injection': 'Web Attack',
        'Infiltration': 'Infiltration',
        'Heartbleed': 'Exploit'
    }
    
    # Notification & Email Settings (Brevo & SMTP)
    BREVO_API_KEY = os.environ.get('BREVO_API_KEY')
    BREVO_SENDER_EMAIL = os.environ.get('BREVO_SENDER_EMAIL') or os.environ.get('GOOGLE_EMAIL', 'teamclickjack@gmail.com')
    BREVO_SENDER_NAME = os.environ.get('BREVO_SENDER_NAME', 'AI-NIDS Security')
    
    SMTP_SERVER = os.environ.get('SMTP_SERVER') or 'smtp.gmail.com'
    SMTP_PORT = int(os.environ.get('SMTP_PORT') or 587)
    SMTP_USERNAME = os.environ.get('SMTP_USERNAME')
    SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD')
    ALERT_EMAIL_RECIPIENTS = os.environ.get('ALERT_EMAIL_RECIPIENTS', '').split(',')
    
    # Webhook Settings
    SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL')
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
    
    # API Settings
    API_RATE_LIMIT = '100 per minute'
    API_KEY_HEADER = 'X-API-Key'
    
    @staticmethod
    def init_app(app):
        """Initialize application with config."""
        # Create necessary directories
        for dir_path in [Config.DATA_DIR, Config.DATASETS_DIR, 
                         Config.PROCESSED_DIR, Config.LOGS_DIR, 
                         Config.MODELS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_ECHO = True
    LOG_LEVEL = 'DEBUG'


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    
    # In production, SECRET_KEY MUST be set via environment variable
    @property
    def SECRET_KEY(self):
        key = os.environ.get('SECRET_KEY')
        if not key:
            raise RuntimeError(
                "SECRET_KEY environment variable is not set. "
                "Set it in your .env file or system environment before running in production."
            )
        return key
    
    # Azure SQL connection string format
    # SQLALCHEMY_DATABASE_URI = 'mssql+pyodbc://user:pass@server.database.windows.net/dbname?driver=ODBC+Driver+17+for+SQL+Server'
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Ensure data directories exist
        import os
        data_dirs = ['data', 'data/logs', 'data/saved_models', 'data/datasets', 'data/processed', 'data/raw']
        for d in data_dirs:
            os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)
        
        # Log to stderr in production
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment."""
    env = os.environ.get('FLASK_ENV') or 'development'
    return config.get(env, config['default'])
