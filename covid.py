import os
from datetime import timedelta

from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import pooling
from mysql.connector import errorcode
from PIL import Image

import numpy as np

# ML imports
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("Warning: TensorFlow not available. Model loading will fail.")
    load_model = None

load_dotenv()

ALLOWED_EXTENSIONS = set((os.getenv("ALLOWED_EXTENSIONS") or "png,jpg,jpeg").split(','))


def parse_input_size(env_value: str):
    try:
        parts = env_value.lower().split('x')
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None


def parse_class_names(env_value: str):
    try:
        names = [n.strip() for n in env_value.split(',') if n.strip()]
        return names if names else ["normal", "covid", "pneumonia"]
    except Exception:
        return ["normal", "covid", "pneumonia"]


def create_app():
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret')
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=6)
    app.config['ASSET_VERSION'] = os.getenv('ASSET_VERSION', '1')

    @app.context_processor
    def inject_asset_version():
        return {'ASSET_VERSION': app.config.get('ASSET_VERSION', '1')}

    upload_folder = os.getenv('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = upload_folder

    # MySQL configuration (lazy pool init)
    dbconfig = {
        'host': os.getenv('MYSQL_HOST', 'localhost'),
        'port': int(os.getenv('MYSQL_PORT', '3306')),
        'user': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', ''),
        'database': os.getenv('MYSQL_DATABASE', 'covid_bd'),
    }
    pool_size = int(os.getenv('MYSQL_POOL_SIZE', '5'))
    connection_pool = None
    schema_migrated = False

    def initialize_database_if_missing(err: Exception):
        try:
            if isinstance(err, mysql.connector.Error) and err.errno == errorcode.ER_BAD_DB_ERROR:
                tmp_config = dict(dbconfig)
                tmp_config.pop('database', None)
                with mysql.connector.connect(**tmp_config) as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{dbconfig['database']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                        conn.commit()
                schema_path = os.path.join(os.getcwd(), 'schema.sql')
                if os.path.exists(schema_path):
                    with mysql.connector.connect(database=dbconfig['database'], **tmp_config) as conn2:
                        with conn2.cursor() as cur2:
                            with open(schema_path, 'r', encoding='utf-8') as f:
                                sql_script = f.read()
                            for statement in [s.strip() for s in sql_script.split(';') if s.strip()]:
                                cur2.execute(statement)
                            conn2.commit()
                return True
        except Exception:
            pass
        return False

    def migrate_schema(conn):
        nonlocal schema_migrated
        if schema_migrated:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                      id INT AUTO_INCREMENT PRIMARY KEY,
                      nom VARCHAR(255) NOT NULL,
                      prenom VARCHAR(255) NOT NULL,
                      email VARCHAR(255) NOT NULL,
                      password VARCHAR(255) NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      UNIQUE KEY uniq_email (email)
                    ) ENGINE=InnoDB
                    """
                )
                cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS prenom VARCHAR(255) NOT NULL AFTER nom")
                cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS password VARCHAR(255) NOT NULL AFTER email")
                cur.execute("ALTER TABLE users ADD UNIQUE KEY IF NOT EXISTS uniq_email (email)")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                      id INT AUTO_INCREMENT PRIMARY KEY,
                      user_id INT NOT NULL,
                      filename VARCHAR(255) NOT NULL,
                      predicted_label VARCHAR(50) NOT NULL,
                      confidence DECIMAL(5,4) NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      INDEX idx_user (user_id),
                      CONSTRAINT fk_predictions_user FOREIGN KEY (user_id)
                        REFERENCES users(id)
                        ON DELETE CASCADE
                    ) ENGINE=InnoDB
                    """
                )
            conn.commit()
            schema_migrated = True
        except Exception:
            pass

    def ensure_pool():
        nonlocal connection_pool
        if connection_pool is None:
            try:
                connection_pool = pooling.MySQLConnectionPool(
                    pool_name="covid_pool", pool_size=pool_size, **dbconfig
                )
                with connection_pool.get_connection() as c:
                    migrate_schema(c)
            except Exception as e:
                if initialize_database_if_missing(e):
                    connection_pool = pooling.MySQLConnectionPool(
                        pool_name="covid_pool", pool_size=pool_size, **dbconfig
                    )
                    with connection_pool.get_connection() as c:
                        migrate_schema(c)
                else:
                    raise RuntimeError(
                        f"MySQL non disponible: {e}. Vérifiez .env (MYSQL_USER/MYSQL_PASSWORD) et que MySQL tourne."
                    )
        return connection_pool

    def get_db_connection():
        pool = ensure_pool()
        return pool.get_connection()

    # Load model once (prefer explicit MODEL_PATH else ADAM model)
    env_model = os.getenv('MODEL_PATH')
    if env_model:
        model_path = env_model
    else:
        model_path = 'meilleur_model_covid_adam.keras'
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}.")
    
    if load_model is None:
        raise RuntimeError("TensorFlow not available. Cannot load model.")
    
    model = load_model(model_path)

    # Determine target input size - Force 128x128 for compatibility
    env_size = parse_input_size(os.getenv('INPUT_SIZE', ''))
    target_size = env_size or (128, 128)
    
    print(f"Using target size: {target_size}")

    class_names_env = os.getenv('CLASS_NAMES', 'normal,covid,pneumonia')
    class_names = parse_class_names(class_names_env)


    def allowed_file(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def preprocess_image(image_path: str, size: tuple[int, int]) -> np.ndarray:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize(size)
            arr = np.array(img).astype('float32') / 255.0
            arr = np.expand_dims(arr, axis=0)
            return arr

    @app.route('/')
    def index():
        if 'user' in session:
            return redirect(url_for('predict'))
        return redirect(url_for('login'))

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            email = request.form.get('email', '').strip().lower()
            nom = request.form.get('nom', '').strip()
            prenom = request.form.get('prenom', '').strip()
            password = request.form.get('password', '')
            if not email or not nom or not prenom or not password:
                flash('Merci de remplir tous les champs.', 'error')
                return redirect(url_for('register'))

            password_hash = generate_password_hash(password)
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO users (nom, prenom, email, password) VALUES (%s, %s, %s, %s)",
                    (nom, prenom, email, password_hash)
                )
                conn.commit()
                cur.close()
                conn.close()
                flash('Inscription réussie. Vous pouvez vous connecter.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                flash(f'Erreur base de données: {e}', 'error')
            finally:
                try:
                    cur.close()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
        return render_template('Register.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            try:
                conn = get_db_connection()
                cur = conn.cursor(dictionary=True)
                cur.execute(
                    "SELECT id, email, nom, prenom, password FROM users WHERE email=%s",
                    (email,)
                )
                user = cur.fetchone()
                cur.close()
                conn.close()
                if user and check_password_hash(user['password'], password):
                    session['user'] = {'id': user['id'], 'email': user['email'], 'nom': user['nom'], 'prenom': user['prenom']}
                    session.permanent = True
                    return redirect(url_for('predict'))
                flash('Identifiants invalides.', 'error')
            except Exception as e:
                flash(f'Erreur base de données: {e}', 'error')
            finally:
                try:
                    cur.close()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
        return render_template('login.html')

    @app.route('/logout')
    def logout():
        session.pop('user', None)
        flash('Déconnecté.', 'info')
        return redirect(url_for('login'))

    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        if 'user' not in session:
            return redirect(url_for('login'))

        prediction_result = None
        if request.method == 'POST':
            if 'image' not in request.files:
                flash('Aucun fichier envoyé.', 'error')
                return redirect(url_for('predict'))
            file = request.files['image']
            if file.filename == '':
                flash('Aucun fichier sélectionné.', 'error')
                return redirect(url_for('predict'))
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                # Try prediction
                try:
                    img_arr = preprocess_image(save_path, target_size)
                    probs = model.predict(img_arr)[0]
                    top_index = int(np.argmax(probs))
                    predicted_label = class_names[top_index] if top_index < len(class_names) else str(top_index)
                    confidence = float(probs[top_index])

                    # save history, auto-create table on 1146
                    try:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        try:
                            cur.execute(
                                "INSERT INTO predictions (user_id, filename, predicted_label, confidence) VALUES (%s, %s, %s, %s)",
                                (session['user']['id'], filename, predicted_label, confidence)
                            )
                        except mysql.connector.Error as db_err:
                            if db_err.errno == errorcode.ER_NO_SUCH_TABLE:
                                cur.execute(
                                    """
                                    CREATE TABLE IF NOT EXISTS predictions (
                                      id INT AUTO_INCREMENT PRIMARY KEY,
                                      user_id INT NOT NULL,
                                      filename VARCHAR(255) NOT NULL,
                                      predicted_label VARCHAR(50) NOT NULL,
                                      confidence DECIMAL(5,4) NOT NULL,
                                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                      INDEX idx_user (user_id),
                                      CONSTRAINT fk_predictions_user FOREIGN KEY (user_id)
                                        REFERENCES users(id)
                                        ON DELETE CASCADE
                                    ) ENGINE=InnoDB
                                    """
                                )
                                cur.execute(
                                    "INSERT INTO predictions (user_id, filename, predicted_label, confidence) VALUES (%s, %s, %s, %s)",
                                    (session['user']['id'], filename, predicted_label, confidence)
                                )
                            else:
                                raise
                        conn.commit()
                        cur.close()
                        conn.close()
                    except Exception as e:
                        flash(f'Impossible d\'enregistrer l\'historique: {e}', 'error')

                    prediction_result = {
                        'label': predicted_label,
                        'confidence': f"{confidence*100:.2f}%"
                    }
                except Exception as e:
                    flash(f'Erreur de prédiction: {e}', 'error')
            else:
                flash('Format de fichier non supporté.', 'error')
        return render_template('Covid.html', prediction=prediction_result)

    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
