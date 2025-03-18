import os
import pickle
import psycopg2
from psycopg2.extras import Json, DictCursor
from datetime import datetime

# PostgreSQL database configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "forecast_models"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "password")
}

def get_db_connection(initialize=True):
    """
    Establishes a connection to the PostgreSQL database and initializes tables if needed.
    
    Parameters:
        initialize (bool): Whether to initialize tables if they don't exist
        
    Returns:
        connection: PostgreSQL database connection
    """
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        
        # Initialize tables if requested
        if initialize:
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'forecast_models'
            );
            """)
            tables_exist = cursor.fetchone()[0]
            
            if not tables_exist:
                print("Tables not found, initializing database...")
                
                # Create table for storing models
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS forecast_models (
                    id SERIAL PRIMARY KEY,
                    company TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    model_binary BYTEA NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    parameters JSONB,
                    last_train_date DATE,
                    metrics JSONB,
                    notes TEXT,
                    version INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE (company, metric, is_active)
                );
                """)
                
                # Create table for model metadata and history
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_history (
                    id SERIAL PRIMARY KEY,
                    model_id INTEGER REFERENCES forecast_models(id),
                    action TEXT NOT NULL,
                    parameters JSONB,
                    metrics JSONB,
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    user_id TEXT,
                    notes TEXT
                );
                """)
                
                conn.commit()
                print("Database initialized successfully")
            
            cursor.close()
        
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def save_model_to_db(model, company, metric, parameters=None, last_train_date=None, metrics=None, notes=None):
    """
    Saves a Prophet model to the database.
    
    Parameters:
        model: The trained Prophet model
        company (str): Company identifier
        metric (str): Metric type (total_deals, house_gross, back_end_gross)
        parameters (dict): Model parameters
        last_train_date (date): The last date used in training
        metrics (dict): Performance metrics
        notes (str): Additional notes
        
    Returns:
        int: The model ID in the database
    """
    try:
        # Serialize the model to a binary object
        model_binary = pickle.dumps(model)
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        # Check if an active model exists for this company and metric
        cursor.execute("""
        SELECT id, version FROM forecast_models 
        WHERE company = %s AND metric = %s AND is_active = TRUE
        """, (company, metric))
        
        existing_model = cursor.fetchone()
        
        if existing_model:
            # Deactivate the existing model
            cursor.execute("""
            UPDATE forecast_models 
            SET is_active = FALSE 
            WHERE id = %s
            """, (existing_model['id'],))
            
            # Calculate new version
            new_version = existing_model['version'] + 1
            
            # Log to history
            cursor.execute("""
            INSERT INTO model_history (model_id, action, parameters, metrics, notes)
            VALUES (%s, %s, %s, %s, %s)
            """, (
                existing_model['id'], 
                'deactivated', 
                Json(parameters) if parameters else None,
                Json(metrics) if metrics else None,
                f"Deactivated due to new version {new_version}"
            ))
        else:
            new_version = 1
        
        # Insert the new model
        cursor.execute("""
        INSERT INTO forecast_models 
        (company, metric, model_binary, parameters, last_train_date, metrics, notes, version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """, (
            company, 
            metric, 
            model_binary, 
            Json(parameters) if parameters else None,
            last_train_date,
            Json(metrics) if metrics else None,
            notes,
            new_version
        ))
        
        model_id = cursor.fetchone()[0]
        
        # Log to history
        cursor.execute("""
        INSERT INTO model_history (model_id, action, parameters, metrics, notes)
        VALUES (%s, %s, %s, %s, %s)
        """, (
            model_id, 
            'created', 
            Json(parameters) if parameters else None,
            Json(metrics) if metrics else None,
            f"New model created, version {new_version}"
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"Model saved to database with ID: {model_id}, version: {new_version}")
        return model_id
        
    except Exception as e:
        print(f"Error saving model to database: {e}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        raise

def load_model_from_db(company, metric):
    """
    Loads a Prophet model from the database.
    
    Parameters:
        company (str): Company identifier
        metric (str): Metric type (total_deals, house_gross, back_end_gross)
        
    Returns:
        tuple: (model, parameters, last_train_date, metrics) or (None, None, None, None) if not found
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute("""
        SELECT model_binary, parameters, last_train_date, metrics
        FROM forecast_models 
        WHERE company = %s AND metric = %s AND is_active = TRUE
        """, (company, metric))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            # Deserialize the model
            model = pickle.loads(result['model_binary'])
            parameters = result['parameters']
            last_train_date = result['last_train_date']
            metrics = result['metrics']
            
            print(f"Model loaded from database for {company} ({metric})")
            return model, parameters, last_train_date, metrics
        else:
            print(f"No active model found for {company} ({metric})")
            return None, None, None, None
            
    except Exception as e:
        print(f"Error loading model from database: {e}")
        if 'conn' in locals() and conn:
            conn.close()
        return None, None, None, None

def delete_model_from_db(company, metric):
    """
    Deactivates a model in the database.
    
    Parameters:
        company (str): Company identifier
        metric (str): Metric type (total_deals, house_gross, back_end_gross)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        UPDATE forecast_models 
        SET is_active = FALSE 
        WHERE company = %s AND metric = %s AND is_active = TRUE
        RETURNING id
        """, (company, metric))
        
        result = cursor.fetchone()
        
        if result:
            model_id = result[0]
            
            # Log to history
            cursor.execute("""
            INSERT INTO model_history (model_id, action, notes)
            VALUES (%s, %s, %s)
            """, (
                model_id, 
                'deleted', 
                f"Model deactivated at {datetime.now().isoformat()}"
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"Model for {company} ({metric}) deactivated successfully")
            return True
        else:
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"No active model found for {company} ({metric})")
            return False
            
    except Exception as e:
        print(f"Error deactivating model: {e}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        return False

def list_models_in_db():
    """
    Lists all active models in the database.
    
    Returns:
        list: List of dictionaries containing model information
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute("""
        SELECT id, company, metric, created_at, updated_at, version
        FROM forecast_models 
        WHERE is_active = TRUE
        ORDER BY company, metric
        """)
        
        models = []
        for row in cursor.fetchall():
            models.append(dict(row))
        
        cursor.close()
        conn.close()
        
        return models
        
    except Exception as e:
        print(f"Error listing models: {e}")
        if 'conn' in locals() and conn:
            conn.close()
        return []