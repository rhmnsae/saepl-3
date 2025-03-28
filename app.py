import os
import re
import sys
from matplotlib.patches import Circle
import pandas as pd
import torch
import nltk
from flask import Flask, current_app, logging, redirect, render_template, request, jsonify, send_file, session, flash, url_for
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing plt
from flask_session import Session
import seaborn as sns
from io import BytesIO
import base64
from collections import Counter
from nltk.corpus import stopwords
import google.generativeai as genai
from datetime import datetime, time
import numpy as np
from nltk.tokenize import word_tokenize
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import CountVectorizer
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.units import inch, cm
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.platypus.flowables import HRFlowable
from reportlab.graphics.charts.barcharts import VerticalBarChart

import secrets
import string
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import jwt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from google.auth.transport import requests as google_requests
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from dotenv import load_dotenv



# Inisialisasi Flask
app = Flask(__name__)
app.secret_key = 'analisis_sentimen_X'  # untuk session management

# Set up server-side sessions to fix the cookie size limit
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_sessions')  # Directory to store sessions
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)  # Create session directory if it doesn't exist
Session(app)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['GOOGLE_CLIENT_ID'] = os.getenv('GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv('GOOGLE_CLIENT_SECRET')
app.config['GOOGLE_REDIRECT_URI'] = os.getenv('GOOGLE_REDIRECT_URI')

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# File untuk menyimpan data pengguna
USERS_FILE = 'users.json'


class User(UserMixin):
    def __init__(self, id, email, password=None, otp=None, otp_expires=None, verified=False):
        self.id = id
        self.email = email
        self.password = password
        self.otp = otp
        self.otp_expires = otp_expires
        self.verified = verified

    @staticmethod
    def get(user_id):
        users = load_users()
        user_data = users.get(user_id)
        if not user_data:
            return None
        return User(
            id=user_id,
            email=user_data.get('email'),
            password=user_data.get('password'),
            otp=user_data.get('otp'),
            otp_expires=user_data.get('otp_expires'),
            verified=user_data.get('verified', False)
        )

    def save(self):
        users = load_users()
        users[self.id] = {
            'email': self.email,
            'password': self.password,
            'otp': self.otp,
            'otp_expires': self.otp_expires,
            'verified': self.verified
        }
        save_users(users)

# Fungsi untuk memuat dan menyimpan data pengguna
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# Fungsi untuk memuat pengguna dari Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# Fungsi untuk menghasilkan OTP
def generate_otp():
    # Menghasilkan OTP 6 digit
    return ''.join(secrets.choice(string.digits) for _ in range(6))

# Fungsi untuk mengirim email OTP
def send_otp_email(email, otp):
    msg = MIMEMultipart()
    msg['From'] = app.config['MAIL_USERNAME']
    msg['To'] = email
    msg['Subject'] = 'Kode OTP untuk Analisis Sentimen X'
    
    body = f'''
    <html>
    <body>
        <h2>Verifikasi Akun Analisis Sentimen X</h2>
        <p>Berikut adalah kode OTP Anda untuk verifikasi:</p>
        <h1 style="color:#4CAF50;">{otp}</h1>
        <p>Kode ini berlaku selama 10 menit.</p>
        <p>Jika Anda tidak meminta kode ini, abaikan email ini.</p>
    </body>
    </html>
    '''
    
    msg.attach(MIMEText(body, 'html'))
    
    try:
        server = smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT'])
        server.starttls()
        server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        text = msg.as_string()
        server.sendmail(app.config['MAIL_USERNAME'], email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Decorator untuk memerlukan verifikasi
def verification_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        
        if not current_user.verified:
            return redirect(url_for('verify_otp'))
            
        return f(*args, **kwargs)
    return decorated_function

# Routes untuk autentikasi
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Validasi input
        if not email or not password:
            flash('Email dan password diperlukan', 'danger')
            return render_template('register.html')
            
        # Cek apakah email sudah terdaftar
        users = load_users()
        for user_id, user_data in users.items():
            if user_data.get('email') == email:
                flash('Email sudah terdaftar', 'danger')
                return render_template('register.html')
        
        # Buat user baru
        user_id = str(len(users) + 1)
        otp = generate_otp()
        otp_expires = (datetime.now() + timedelta(minutes=10)).timestamp()
        
        new_user = User(
            id=user_id,
            email=email,
            password=generate_password_hash(password),
            otp=otp,
            otp_expires=otp_expires,
            verified=False
        )
        new_user.save()
        
        # Kirim OTP ke email
        if send_otp_email(email, otp):
            # Login user
            login_user(new_user)
            flash('Pendaftaran berhasil. Silakan verifikasi akun Anda.', 'success')
            return redirect(url_for('verify_otp'))
        else:
            flash('Gagal mengirim OTP. Silakan coba lagi.', 'danger')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Validasi input
        if not email or not password:
            flash('Email dan password diperlukan', 'danger')
            return render_template('login.html')
        
        # Cari user dengan email
        users = load_users()
        user_id = None
        for uid, user_data in users.items():
            if user_data.get('email') == email:
                user_id = uid
                break
        
        if not user_id:
            flash('Email atau password salah', 'danger')
            return render_template('login.html')
        
        user = User.get(user_id)
        
        # Verifikasi password
        if check_password_hash(user.password, password):
            login_user(user)
            
            # Jika belum terverifikasi, kirim OTP baru
            if not user.verified:
                otp = generate_otp()
                user.otp = otp
                user.otp_expires = (datetime.now() + timedelta(minutes=10)).timestamp()
                user.save()
                
                if send_otp_email(email, otp):
                    return redirect(url_for('verify_otp'))
                else:
                    flash('Gagal mengirim OTP. Silakan coba lagi.', 'danger')
                    return render_template('login.html')
            
            return redirect(url_for('index'))
        else:
            flash('Email atau password salah', 'danger')
    
    return render_template('login.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
@login_required
def verify_otp():
    if current_user.verified:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        otp = request.form.get('otp')
        
        # Validasi OTP
        if not otp:
            flash('OTP diperlukan', 'danger')
            return render_template('verify_otp.html')
        
        if current_user.otp == otp:
            # Cek apakah OTP masih berlaku
            now = datetime.now().timestamp()
            if now > float(current_user.otp_expires):
                flash('OTP sudah kedaluwarsa. Silakan minta OTP baru.', 'danger')
                return render_template('verify_otp.html')
                
            # Verifikasi user
            current_user.verified = True
            current_user.otp = None
            current_user.otp_expires = None
            current_user.save()
            
            flash('Akun berhasil diverifikasi', 'success')
            return redirect(url_for('index'))
        else:
            flash('OTP tidak valid', 'danger')
    
    return render_template('verify_otp.html')

@app.route('/resend_otp', methods=['POST'])
@login_required
def resend_otp():
    if current_user.verified:
        return redirect(url_for('index'))
        
    # Generate OTP baru
    otp = generate_otp()
    current_user.otp = otp
    current_user.otp_expires = (datetime.now() + timedelta(minutes=10)).timestamp()
    current_user.save()
    
    # Kirim OTP ke email
    if send_otp_email(current_user.email, otp):
        flash('OTP baru telah dikirim ke email Anda', 'success')
    else:
        flash('Gagal mengirim OTP. Silakan coba lagi.', 'danger')
    
    return redirect(url_for('verify_otp'))

@app.route('/auth/google')
def auth_google():
    # Buat flow object untuk autentikasi Google
    flow = Flow.from_client_secrets_file(
        'client_secrets.json',
        scopes=['openid', 'email', 'profile'],
        redirect_uri=app.config['GOOGLE_REDIRECT_URI']
    )
    
    # Hasilkan URL untuk autentikasi Google
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
    )
    
    # Simpan state di session
    session['google_state'] = state
    
    return redirect(authorization_url)

@app.route('/callback')
def callback():
    # Verifikasi state untuk mencegah CSRF
    state = session.get('google_state')
    if not state or request.args.get('state') != state:
        flash('Autentikasi tidak valid', 'danger')
        return redirect(url_for('login'))
    
    # Ambil flow dari state
    flow = Flow.from_client_secrets_file(
        'client_secrets.json',
        scopes=['openid', 'email', 'profile'],
        redirect_uri=app.config['GOOGLE_REDIRECT_URI'],
        state=state
    )
    
    # Minta token access
    flow.fetch_token(authorization_response=request.url)
    
    # Dapatkan info pengguna dari ID token
    id_info = id_token.verify_oauth2_token(
        flow.credentials.id_token,
        google_requests.Request(),
        app.config['GOOGLE_CLIENT_ID']
    )
    
    # Ambil email dari info pengguna
    email = id_info.get('email')
    
    # Cek apakah email sudah diverifikasi Google
    if not id_info.get('email_verified'):
        flash('Email Google Anda belum diverifikasi', 'danger')
        return redirect(url_for('login'))
    
    # Cek apakah pengguna sudah ada
    users = load_users()
    user_id = None
    for uid, user_data in users.items():
        if user_data.get('email') == email:
            user_id = uid
            break
    
    if user_id:
        # Jika pengguna sudah ada, login
        user = User.get(user_id)
    else:
        # Jika pengguna belum ada, buat pengguna baru
        user_id = str(len(users) + 1)
        user = User(
            id=user_id,
            email=email,
            verified=True  # Pengguna Google otomatis terverifikasi
        )
        user.save()
    
    # Login pengguna
    login_user(user)
    
    return redirect(url_for('index'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Anda telah keluar', 'info')
    return redirect(url_for('login'))



# Konfigurasi folder untuk upload file
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Konfigurasi folder untuk model
MODEL_FOLDER = 'models'
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Definisikan path model
MODEL_PATH = os.path.join(MODEL_FOLDER, 'indobert_sentiment_best.pt')

# Konfigurasi Google Gemini API
GEMINI_API_KEY = "AIzaSyCYPhQCDxpyz_MmR86v43XgKvMryx5FfQY"  # Ganti dengan API key Anda
genai.configure(api_key=GEMINI_API_KEY)

# Cek keberadaan model pada startup
if not os.path.exists(MODEL_PATH):
    print(f"PERINGATAN: Model {MODEL_PATH} tidak ditemukan. Pastikan model sudah ada di folder models/")

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ===================
# Fungsi Preprocessing 
# ===================

def preprocess_text(text):
    """
    Preprocessing teks untuk analisis sentimen
    """
    if pd.isna(text):
        return ""
        
    # Konversi ke string jika bukan string
    if not isinstance(text, str):
        text = str(text)
    
    # Menghapus URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Menghapus mentions
    text = re.sub(r'@\w+', '', text)
    
    # Menghapus hashtag (kita hanya hapus tanda #, kata tetap dipertahankan)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Menghapus karakter khusus dan angka
    text = re.sub(r'[^\w\s]', '', text)
    
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """
    Tokenize text into words and remove stopwords
    """
    if pd.isna(text) or text == "":
        return []
    
    # Download stopwords if needed
    try:
        id_stopwords = set(stopwords.words('indonesian'))
    except:
        nltk.download('stopwords')
        id_stopwords = set(stopwords.words('indonesian'))
    
    # Add custom stopwords
    custom_stopwords = {
        'yang', 'dan', 'di', 'dengan', 'untuk', 'pada', 'dalam', 'adalah', 'ini', 'itu',
        'ada', 'akan', 'dari', 'ke', 'ya', 'juga', 'saya', 'kita', 'kami', 'mereka',
        'dia', 'anda', 'atau', 'bahwa', 'karena', 'oleh', 'jika', 'maka', 'masih', 'dapat',
        'bisa', 'tersebut', 'agar', 'sebagai', 'secara', 'seperti', 'hingga', 'telah', 'tidak',
        'tak', 'tanpa', 'tapi', 'tetapi', 'lalu', 'mau', 'harus', 'namun', 'ketika', 'saat',
        'http', 'https', 'co', 't', 'a', 'amp', 'rt', 'nya', 'yg', 'dgn', 'utk', 'dr',
        'pd', 'jd', 'sdh', 'tdk', 'bgt', 'kalo', 'gitu', 'gak', 'kan', 'deh', 'sih'
    }
    
    all_stopwords = id_stopwords.union(custom_stopwords)
    
    # Simple word tokenization without relying on NLTK's word_tokenize
    # Split by whitespace and filter out non-alphanumeric characters
    try:
        # Try to use NLTK's word_tokenize if available
        nltk.download('punkt')
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text.lower())
    except:
        # Fallback to simple tokenization if NLTK's word_tokenize fails
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    
    # Remove stopwords and short words
    filtered_tokens = [word for word in tokens if word not in all_stopwords and len(word) > 2]
    
    return filtered_tokens

# ===================
# Fungsi Analisis Sentimen
# ===================

def load_sentiment_model(model_name='indolem/indobert-base-uncased'):
    """
    Memuat model IndoBERT dan tokenizer
    """
    # Cek apakah model terlatih ada
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model {MODEL_PATH} tidak ditemukan. Pastikan model terlatih tersedia.")
    
    # Muat tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Muat model dasar
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3  # 3 kelas: Negatif (0), Netral (1), Positif (2)
    )
    
    # Muat state model yang terlatih
    print(f"Memuat model terlatih dari {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    
    return tokenizer, model

def predict_sentiments(file_path):
    """
    Memprediksi sentimen dari teks dalam file CSV menggunakan model IndoBERT terlatih
    """
    # Muat data
    df = pd.read_csv(file_path)
    
    # Pastikan kolom full_text atau text ada
    if 'full_text' not in df.columns and 'text' not in df.columns:
        raise ValueError("File CSV harus memiliki kolom 'full_text' atau 'text'")
    
    # Gunakan kolom text jika full_text tidak ada
    text_column = 'full_text' if 'full_text' in df.columns else 'text'
    
    # Preprocessing teks
    print("Preprocessing teks...")
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Muat tokenizer dan model terlatih
    tokenizer, model = load_sentiment_model()
    
    # Siapkan device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Siapkan hasil
    results = []
    confidences = []
    sentiment_labels = ['Negatif', 'Netral', 'Positif']
    
    print(f"Melakukan prediksi sentimen untuk {len(df)} tweets...")
    
    # Prediksi dalam batch
    batch_size = 16
    for i in range(0, len(df), batch_size):
        batch_texts = df['processed_text'].iloc[i:i+batch_size].tolist()
        
        # Tokenisasi
        encodings = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Prediksi
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            batch_results = [sentiment_labels[pred] for pred in preds.cpu().tolist()]
            batch_confidences = [float(probs[i, preds[i]]) * 100 for i in range(len(preds))]
            
            results.extend(batch_results)
            confidences.extend(batch_confidences)
    
    # Tambahkan hasil prediksi ke dataframe
    df['predicted_sentiment'] = results
    df['confidence'] = confidences
    
    # Tambahkan tanggal jika ada created_at
    if 'created_at' in df.columns:
        df['date'] = pd.to_datetime(df['created_at'], 
                             format='%a %b %d %H:%M:%S %z %Y',  # Format Twitter
                             errors='coerce').dt.strftime('%d %b %Y')
    else:
        # Gunakan tanggal hari ini jika tidak ada kolom created_at
        df['date'] = datetime.now().strftime('%d %b %Y')
    
    # Tambahkan kolom untuk likes, retweets, dan replies jika tidak ada
    if 'favorite_count' not in df.columns:
        df['favorite_count'] = 0
    if 'retweet_count' not in df.columns:
        df['retweet_count'] = 0
    if 'reply_count' not in df.columns:
        df['reply_count'] = 0
    
    # Ganti nama kolom untuk konsistensi dalam UI
    rename_columns = {
        'screen_name': 'username',
        'favorite_count': 'likes',
        'retweet_count': 'retweets',
        'reply_count': 'replies',
        text_column: 'content'
    }
    df = df.rename(columns={col: new_col for col, new_col in rename_columns.items() if col in df.columns})
    
    # Pastikan semua kolom yang diperlukan ada
    required_cols = ['username', 'content', 'date', 'likes', 'retweets', 'replies', 'predicted_sentiment', 'confidence']
    for col in required_cols:
        if col not in df.columns:
            if col == 'username':
                df['username'] = 'user' + df.index.astype(str)
            else:
                df[col] = 0 if col in ['likes', 'retweets', 'replies', 'confidence'] else ''
    
    # Ensure tweet URL is available
    if 'tweet_url' not in df.columns and 'id_str' in df.columns:
        df['tweet_url'] = 'https://twitter.com/i/web/status/' + df['id_str'].astype(str)
    
    # Ensure image URL is available
    if 'image_url' not in df.columns and 'media_url' in df.columns:
        df['image_url'] = df['media_url']
    
    print("Prediksi selesai.")
    return df

def extract_hashtags(df):
    """
    Ekstrak hashtag dari tweets dan hitung frekuensinya
    """
    hashtag_pattern = re.compile(r'#(\w+)')
    all_hashtags = []
    
    for text in df['content']:
        try:
            hashtags = hashtag_pattern.findall(str(text).lower())
            all_hashtags.extend(hashtags)
        except:
            continue
    
    hashtag_counts = Counter(all_hashtags)
    return hashtag_counts

def extract_topics(df, num_topics=10, min_count=3):
    """
    Ekstrak topik dari dataset berdasarkan frekuensi kata
    """
    # Download stopwords jika belum ada
    try:
        indonesian_stopwords = set(stopwords.words('indonesian'))
    except:
        nltk.download('stopwords')
        indonesian_stopwords = set(stopwords.words('indonesian'))
    
    # Tambahkan stopwords kustom
    custom_stopwords = {
        'yang', 'dan', 'di', 'dengan', 'untuk', 'pada', 'dalam', 'dari', 
        'ke', 'ya', 'ini', 'itu', 'ada', 'juga', 'saya', 'kita', 'akan'
    }
    all_stopwords = indonesian_stopwords.union(custom_stopwords)
    
    # Ekstrak kata-kata dan hitung frekuensi
    word_freq = Counter()
    
    for text in df['processed_text']:
        try:
            words = str(text).lower().split()
            # Filter stopwords dan kata pendek
            filtered_words = [word for word in words if word not in all_stopwords and len(word) > 3]
            word_freq.update(filtered_words)
        except:
            continue
    
    # Ekstrak topik (kata-kata dengan frekuensi tertinggi)
    topics = [{"topic": word, "frequency": count} 
              for word, count in word_freq.most_common(num_topics) 
              if count >= min_count]
    
    return topics

def analyze_sentiment_per_hashtag(df):
    """
    Menganalisis sentimen berdasarkan hashtag
    """
    hashtag_pattern = re.compile(r'#(\w+)')
    hashtag_sentiments = {}
    
    for _, row in df.iterrows():
        try:
            text = str(row['content']).lower()
            sentiment = row['predicted_sentiment']
            hashtags = hashtag_pattern.findall(text)
            
            for tag in hashtags:
                if tag not in hashtag_sentiments:
                    hashtag_sentiments[tag] = {'Positif': 0, 'Netral': 0, 'Negatif': 0, 'total': 0}
                
                hashtag_sentiments[tag][sentiment] += 1
                hashtag_sentiments[tag]['total'] += 1
        except:
            continue
    
    # Convert to percentage and format for display
    result = []
    for tag, counts in hashtag_sentiments.items():
        if counts['total'] >= 3:  # Only include hashtags that appear at least 3 times
            result.append({
                'tag': f'#{tag}',
                'positive': round(counts['Positif'] / counts['total'] * 100),
                'neutral': round(counts['Netral'] / counts['total'] * 100),
                'negative': round(counts['Negatif'] / counts['total'] * 100),
                'total': counts['total']
            })
    
    return sorted(result, key=lambda x: x['total'], reverse=True)[:10]  # Return top 10

def get_top_users(df):
    """
    Mendapatkan pengguna dengan tweet terbanyak dan sentimen dominan mereka
    """
    if 'username' not in df.columns:
        return []
        
    # Group by username and count tweets
    user_counts = df.groupby('username').size().reset_index(name='count')
    
    # Calculate average engagement
    engagement_df = df.groupby('username')[['likes', 'retweets', 'replies']].mean().sum(axis=1).reset_index(name='avg_engagement')
    user_counts = user_counts.merge(engagement_df, on='username')
    
    # Get dominant sentiment for each user
    sentiment_counts = df.groupby(['username', 'predicted_sentiment']).size().reset_index(name='sentiment_count')
    dominant_sentiment = sentiment_counts.loc[sentiment_counts.groupby('username')['sentiment_count'].idxmax()]
    dominant_sentiment = dominant_sentiment[['username', 'predicted_sentiment']].rename(columns={'predicted_sentiment': 'dominant_sentiment'})
    
    # Merge all information
    user_info = user_counts.merge(dominant_sentiment, on='username')
    
    # Sort by tweet count and return top 10
    user_info = user_info.sort_values('count', ascending=False).head(10)
    
    # Convert to list of dictionaries for JSON
    return user_info.to_dict('records')


# Download all necessary NLTK resources
def download_nltk_resources():
    """Download all necessary NLTK resources"""
    try:
        # Download essential NLTK packages
        nltk.download('stopwords')
        nltk.download('punkt')
        print("NLTK resources downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        
# Call this function at the start
download_nltk_resources()

def extract_words_by_sentiment(df):
    """
    Extract most frequent words for each sentiment category with improved processing
    """
    # Initialize word frequency counters
    positive_words = Counter()
    neutral_words = Counter()
    negative_words = Counter()
    
    # Expanded stopwords with comprehensive list
    stopwords_list = set([
        'yang', 'dan', 'di', 'dengan', 'untuk', 'pada', 'dalam', 'adalah', 'ini', 'itu',
        'ada', 'akan', 'dari', 'ke', 'ya', 'juga', 'saya', 'kita', 'kami', 'mereka',
        'dia', 'anda', 'atau', 'bahwa', 'karena', 'oleh', 'jika', 'maka', 'masih', 'dapat',
        'bisa', 'tersebut', 'agar', 'sebagai', 'secara', 'seperti', 'hingga', 'telah', 'tidak',
        'tak', 'tanpa', 'tapi', 'tetapi', 'lalu', 'mau', 'harus', 'namun', 'ketika', 'saat',
        'http', 'https', 'co', 't', 'a', 'amp', 'rt', 'nya', 'yg', 'dgn', 'utk', 'dr',
        'pd', 'jd', 'sdh', 'tdk', 'bgt', 'kalo', 'gitu', 'gak', 'kan', 'deh', 'sih',
        'nih', 'si', 'oh', 'udah', 'udh', 'eh', 'ah', 'lah', 'ku', 'mu', 'nya', 'ni',
        'aja', 'dg', 'lg', 'yah', 'ya', 'ga', 'gk', 'kk', 'jg', 'sy', 'dpt', 'dtg',
        'bnr', 'tp', 'krn', 'kpd', 'jgn', 'cm', 'blm', 'sdg', 'skrg', 'ckp', 'cuma'
    ])
    
    try:
        # Process each tweet with more robust handling
        for _, row in df.iterrows():
            if pd.isna(row.get('processed_text', '')) or row.get('processed_text', '') == "":
                continue
                
            # Get sentiment
            sentiment = row.get('predicted_sentiment')
            if not sentiment:
                continue
                
            # Better tokenization with preprocessing
            text = row.get('processed_text', '').lower()
            
            # Remove URLs
            text = re.sub(r'https?:\/\/\S+', '', text)
            
            # Remove mentions and hashtags symbols but keep the words
            text = re.sub(r'@(\w+)', r'\1', text)
            text = re.sub(r'#(\w+)', r'\1', text)
            
            # Remove special characters and digits
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize
            tokens = text.split()
            
            # Filter out stopwords and short words
            filtered_tokens = [word for word in tokens 
                             if word not in stopwords_list 
                             and len(word) > 3
                             and not word.isdigit()]
            
            # Add to appropriate counter based on sentiment
            if sentiment == 'Positif':
                positive_words.update(filtered_tokens)
            elif sentiment == 'Netral':
                neutral_words.update(filtered_tokens)
            elif sentiment == 'Negatif':
                negative_words.update(filtered_tokens)
        
        # Get top words for each sentiment (top 20 to ensure we have enough for visualization)
        return {
            'positive': [{"word": word, "count": count} for word, count in positive_words.most_common(20)],
            'neutral': [{"word": word, "count": count} for word, count in neutral_words.most_common(20)],
            'negative': [{"word": word, "count": count} for word, count in negative_words.most_common(20)]
        }
    except Exception as e:
        print(f"Error in extract_words_by_sentiment: {e}")
        # Return empty data if there's an error
        return {
            'positive': [],
            'neutral': [],
            'negative': []
        }

# Update the create_sentiment_plot function with the new color scheme
def create_sentiment_plot(df):
    """
    Creates minimalist aesthetic sentiment distribution plot
    with consistent style matching word frequency visualization
    """
    # Tetapkan gaya seaborn yang minimalis
    sns.set_style("whitegrid", {'grid.linestyle': ':'})
    
    plt.figure(figsize=(5, 3))
    
    # Urutan sentimen yang konsisten
    sentiment_order = ['Positif', 'Netral', 'Negatif']
    
    # Warna monokrom yang sesuai dengan visualisasi frekuensi kata
    color_map = {
        'Positif': '#ffffff',  # Abu-abu sedang
        'Netral': '#9e9e9e',   # Abu-abu terang
        'Negatif': '#000000'   # Abu-abu gelap
    }
    
    # Buat plot dengan warna yang sesuai
    ax = sns.countplot(
        data=df, 
        x='predicted_sentiment', 
        palette=color_map,
        order=sentiment_order,
        edgecolor='black',
        alpha=0.9,
    )
    
    # Hapus bingkai kecuali di bagian bawah
    sns.despine(left=True, bottom=False)
    
    # Judul dan label sederhana
    plt.title('Distribusi Sentimen pada Dataset', fontsize=8, pad=6, fontweight='bold')
    plt.xlabel('Sentimen', fontsize=6)
    plt.ylabel('Jumlah', fontsize=6)
    
    # Rotasi xticks untuk menghemat ruang
    plt.xticks(fontsize=6)
    
    # Angka di atas bar dengan ukuran yang sesuai
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2., 
            height + 5, 
            f'{int(height)}', 
            ha='center', 
            va='bottom', 
            fontsize=7
        )
    
    # Atur jarak antar komponen
    plt.tight_layout(pad=1.0)
    
    # Simpan plot ke buffer dan encode ke base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_data

def create_improved_word_cloud(df):
    """
    Creates enhanced word cloud from tweets with better visualization and optimization
    with larger text display and improved overall appearance
    """
    try:
        if 'processed_text' not in df.columns or len(df) == 0:
            return None
        
        # Combine all processed text
        all_text = ' '.join(df['processed_text'].dropna().astype(str).tolist())
        
        # Enhanced stopwords for cleaner visualization
        extra_stopwords = {
            'yang', 'dan', 'di', 'dengan', 'untuk', 'pada', 'dalam', 'adalah', 'ini', 'itu',
            'ada', 'akan', 'dari', 'ke', 'ya', 'juga', 'rt', 'amp', 'yg', 'dgn', 'utk', 'dr',
            'pd', 'jd', 'sdh', 'tdk', 'bisa', 'ada', 'kalo', 'bgt', 'aja', 'gitu', 'gak', 'mau',
            'biar', 'kan', 'klo', 'deh', 'sih', 'nya', 'nih', 'loh', 'juga', 'kita', 'kami',
            'saya', 'mereka', 'dia', 'anda', 'atau', 'bahwa', 'karena', 'oleh', 'jika', 'maka',
            'masih', 'dapat', 'tersebut', 'agar', 'sebagai', 'secara', 'seperti', 'hingga', 'si',
            'oh', 'udah', 'udh', 'eh', 'ah', 'lah', 'ku', 'mu', 'ni', 'aja', 'dg', 'lg', 'yah',
            'ga', 'gk', 'kk', 'jg', 'sy', 'krn', 'tp', 'trs', 'dr', 'kl', 'bs', 'sm', 'dpt',
            'dtg', 'bnr', 'kpd', 'jgn', 'cm', 'blm', 'sdg', 'skrg', 'ckp', 'cuma'
        }
        
        # Tokenize text more effectively
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        
        # Count word frequencies excluding stopwords
        word_freq = Counter([word for word in words if word not in extra_stopwords])
        
        # Get top words for visualization
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:150])
        
        if not top_words:
            return None
        
        # Increase figure size for better proportions and text readability
        plt.figure(figsize=(40, 25))
        
        # Use only top 30 words for better readability with larger text
        words = list(top_words.keys())[:100]
        frequencies = list(top_words.values())[:100]
        y_pos = np.arange(len(words))
        
        # Create color gradient from dark blue to light blue for better contrast
        colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(y_pos)))
        
        # Create horizontal bar chart with enhanced styling and thicker bars
        bars = plt.barh(y_pos, frequencies, color=colors, height=0.9)
        
        # EXTREMELY LARGE TEXT: Dramatically increase font sizes for all text elements
        plt.yticks(y_pos, words, fontsize=50, fontweight='bold')  # Much larger word labels
        plt.xticks(fontsize=46)  # Much larger x-axis values
        plt.xlabel('Frekuensi', fontsize=56, fontweight='bold')  # Much larger x-axis label
        plt.title('Kata yang Sering Muncul dalam Tweet', fontsize=64, fontweight='bold')  # Much larger title
        
        # Add value labels to bars with increased padding and much larger font size
        for i, v in enumerate(frequencies):
            plt.text(v + (max(frequencies) * 0.02), i, str(v), 
                    color='black', fontweight='bold', fontsize=52,
                    va='center')
        
        # Improve overall appearance
        plt.grid(axis='x', linestyle='--', alpha=0.2)  # Subtle grid
        plt.tick_params(axis='both', which='major', pad=10)  # Add padding around ticks
        plt.gca().spines['top'].set_visible(False)  # Remove top border
        plt.gca().spines['right'].set_visible(False)  # Remove right border
        plt.gca().spines['bottom'].set_linewidth(2)  # Make bottom border thicker
        plt.gca().spines['left'].set_linewidth(2)  # Make left border thicker
        
        # Add more space to accommodate larger text
        plt.tight_layout(pad=4.0)
        
        # Create a cleaner, more professional visualization with higher resolution
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=500, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_data
    except Exception as e:
        print(f"Error creating improved word cloud: {e}")
        return None
    
# ===================
# Fungsi Chatbot dengan Google Gemini
# ===================

def query_gemini(prompt, analysis_context=None):
    """
    Mengirim pertanyaan ke Google Gemini API dan mendapatkan respons
    """
    try:
        # Siapkan model
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        
        # Tambahkan konteks analisis sentimen jika tersedia
        if analysis_context:
            context = f"""
            Berikut adalah hasil analisis sentimen dari data X tentang {analysis_context['title']}:
            - Total tweet: {analysis_context['total_tweets']}
            - Sentimen Positif: {analysis_context['positive_count']} tweets ({analysis_context['positive_percent']}%)
            - Sentimen Netral: {analysis_context['neutral_count']} tweets ({analysis_context['neutral_percent']}%)
            - Sentimen Negatif: {analysis_context['negative_count']} tweets ({analysis_context['negative_percent']}%)
            
            Topik utama yang dibicarakan: {', '.join(analysis_context['top_topics'])}
            
            Hashtag populer: {', '.join(analysis_context['top_hashtags'])}
            
            Judul analisis: {analysis_context['title']}
            
            Berdasarkan data tersebut, silakan berikan evaluasi kebijakan dan respons untuk pertanyaan berikut.
            Jawab dalam bahasa Indonesia yang formal dan profesional, namun tetap mudah dimengerti.
            Cantumkan angka-angka penting dari data analisis untuk mendukung argumentasi.
            Berikan 2-3 rekomendasi spesifik untuk perbaikan kebijakan berdasarkan sentimen publik yang terdeteksi.
            
            Format respons dengan paragraf yang terpisah, gunakan tanda '*' untuk kata yang perlu ditekankan, 
            dan buat daftar dengan tanda '-' jika diperlukan.
            """
            
            # Detect if prompt is about a specific policy
            
            full_prompt = f"{context}\n\nPertanyaan pengguna: {prompt}\n\n"
        else:
            full_prompt = prompt
        
        # Set generation config
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Send request to Gemini
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        return response.text
    except Exception as e:
        return f"Maaf, terjadi kesalahan dalam berkomunikasi dengan Gemini API: {str(e)}"

# ===================
# Routes Flask
# ===================

@app.route('/')
@verification_required
def index():
    # Cek model terlatih
    if not os.path.exists(MODEL_PATH):
        flash("PERINGATAN: Model terlatih tidak ditemukan di path models/indobert_sentiment_best.pt. Aplikasi mungkin tidak berfungsi dengan benar.", "warning")
    return render_template('index.html')


# Update the upload_file route to include improved visualizations
@app.route('/upload', methods=['POST'])
def upload_file():
    # Verifikasi model terlatih ada
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': f'Model terlatih tidak ditemukan: {MODEL_PATH}. Mohon pindahkan model Anda ke folder models/'})
    
    if 'csv-file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['csv-file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Simpan hasil analisis ke file
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], f'analyzed_{filename}')
            
            # Proses file dan lakukan analisis sentimen
            result_df = predict_sentiments(file_path)
            
            # Simpan hasil analisis untuk penggunaan selanjutnya
            result_df.to_csv(output_file, index=False)
            
            # Simpan ONLY THE PATH to the analysis file in session, not the entire results
            session['analysis_file'] = output_file
            
            # Hitung statistik
            sentiment_counts = result_df['predicted_sentiment'].value_counts()
            total_tweets = len(result_df)
            
            # Ekstrak hashtags
            hashtag_counts = extract_hashtags(result_df)
            
            # Ekstrak topik
            topics = extract_topics(result_df)
            
            # Analisis sentimen per hashtag
            hashtag_sentiment = analyze_sentiment_per_hashtag(result_df)
            
            # Dapatkan pengguna teratas
            top_users = get_top_users(result_df)
            
            # Try to extract words by sentiment with improved function
            try:
                sentiment_words = extract_words_by_sentiment(result_df)
            except Exception as e:
                print(f"Error in sentiment word extraction: {e}")
                sentiment_words = {
                    'positive': [],
                    'neutral': [],
                    'negative': []
                }
            
            # Create plots with updated color scheme
            sentiment_plot = create_sentiment_plot(result_df)
            
            # Create improved word cloud with new function
            try:
                word_cloud = create_improved_word_cloud(result_df)
            except Exception as e:
                print(f"Error creating word cloud: {e}")
                word_cloud = None
            
            # Siapkan data untuk tampilan
            positive_count = int(sentiment_counts.get('Positif', 0))
            neutral_count = int(sentiment_counts.get('Netral', 0))
            negative_count = int(sentiment_counts.get('Negatif', 0))
            
            title = request.form.get('title', 'Analisis Sentimen X')
            description = request.form.get('description', '')
            
            # Create analysis results with error checking for all components
            analysis_results = {
                'title': title,
                'description': description,
                'total_tweets': total_tweets,
                'positive_count': positive_count,
                'neutral_count': neutral_count,
                'negative_count': negative_count,
                'positive_percent': round((positive_count / total_tweets * 100), 1) if total_tweets > 0 else 0,
                'neutral_percent': round((neutral_count / total_tweets * 100), 1) if total_tweets > 0 else 0,
                'negative_percent': round((negative_count / total_tweets * 100), 1) if total_tweets > 0 else 0,
                'top_hashtags': [{'tag': tag, 'count': count} for tag, count in hashtag_counts.most_common(10)],
                'topics': topics,
                'hashtag_sentiment': hashtag_sentiment,
                'top_users': top_users,
                'sentiment_words': {
                    'positive': sentiment_words.get('positive', []),
                    'neutral': sentiment_words.get('neutral', []),
                    'negative': sentiment_words.get('negative', [])
                },
                'sentiment_plot': sentiment_plot,
                'word_cloud': word_cloud if word_cloud else None
            }
            
            # Add necessary fields to tweets that we'll send directly to the client
            tweets_for_display = []
            for _, row in result_df.iterrows():
                tweet = {
                    'username': row.get('username', ''),
                    'content': row.get('content', ''),
                    'date': row.get('date', ''),
                    'likes': row.get('likes', 0),
                    'retweets': row.get('retweets', 0),
                    'replies': row.get('replies', 0),
                    'predicted_sentiment': row.get('predicted_sentiment', ''),
                    'confidence': row.get('confidence', 0)
                }
                
                # Add optional fields if they exist
                for field in ['tweet_url', 'image_url', 'lang', 'location']:
                    if field in row and not pd.isna(row[field]):
                        tweet[field] = row[field]
                
                tweets_for_display.append(tweet)
            
            # Add tweets to the results
            analysis_results['tweets'] = tweets_for_display
            
            # Store minimal context for chatbot in session, not the entire results
            session['analysis_context'] = {
                'title': title,
                'description': description,
                'total_tweets': total_tweets,
                'positive_count': positive_count,
                'neutral_count': neutral_count, 
                'negative_count': negative_count,
                'positive_percent': analysis_results['positive_percent'],
                'neutral_percent': analysis_results['neutral_percent'],
                'negative_percent': analysis_results['negative_percent'],
                'top_hashtags': [h['tag'] for h in analysis_results['top_hashtags'][:5]],
                'top_topics': [t['topic'] for t in topics[:5]]
            }
            
            return jsonify(analysis_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)})

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    message = data.get('message', '')
    
    # Get the analysis context from session
    analysis_context = session.get('analysis_context')
    
    # Send message to Gemini and get response
    response = query_gemini(message, analysis_context)
    
    return jsonify({'response': response})

# Update the filter_tweets route to read from the file
@app.route('/filter_tweets', methods=['POST'])
def filter_tweets():
    data = request.json
    sentiment_filter = data.get('sentiment', 'all')
    
    # Get the analysis file path from session
    file_path = session.get('analysis_file')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No analysis data available'})
    
    # Read the analysis results from file
    result_df = pd.read_csv(file_path)
    
    # Filter by sentiment
    if sentiment_filter != 'all':
        result_df = result_df[result_df['predicted_sentiment'] == sentiment_filter]
    
    # Prepare tweets for display
    tweets_for_display = []
    for _, row in result_df.iterrows():
        tweet = {
            'username': row.get('username', ''),
            'content': row.get('content', ''),
            'date': row.get('date', ''),
            'likes': row.get('likes', 0),
            'retweets': row.get('retweets', 0),
            'replies': row.get('replies', 0),
            'predicted_sentiment': row.get('predicted_sentiment', ''),
            'confidence': row.get('confidence', 0)
        }
        
        # Add optional fields if they exist
        for field in ['tweet_url', 'image_url', 'lang', 'location']:
            if field in row and not pd.isna(row[field]):
                tweet[field] = row[field]
        
        tweets_for_display.append(tweet)
    
    return jsonify({'tweets': tweets_for_display})


@app.route('/download_report', methods=['GET'])
def download_report():
    # Cek apakah ada data analisis di session
    if 'analysis_file' not in session or 'analysis_context' not in session:
        flash("Tidak ada data analisis yang tersedia. Silakan upload file CSV terlebih dahulu.", "warning")
        return redirect(url_for('index'))
    
    # Ambil data analisis dari file
    file_path = session.get('analysis_file')
    if not os.path.exists(file_path):
        flash("File analisis tidak ditemukan.", "error")
        return redirect(url_for('index'))
    
    # Baca data analisis
    analysis_df = pd.read_csv(file_path)
    
    # Ambil konteks analisis dari session
    analysis_context = session.get('analysis_context')
    
    # Siapkan buffer untuk PDF
    buffer = io.BytesIO()
    
    # Buat dokumen PDF
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           rightMargin=50, leftMargin=50,
                           topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    elements = []
    
    # ============= DEFINISI STYLE =============
    
    # Warna
    primary_color = colors.HexColor('#1b3a59')     # Biru tua
    secondary_color = colors.HexColor('#2d6b9c')   # Biru sedang
    accent_color = colors.HexColor('#f1b211')      # Kuning emas
    light_color = colors.HexColor('#e6e6e6')       # Abu muda
    
    # Definisi style halaman
    def add_page_number(canvas, doc):
        page_num = canvas.getPageNumber()
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.darkgrey)
        canvas.drawRightString(doc.pagesize[0] - 50, 30, f"Halaman {page_num}")
        canvas.drawString(50, 30, f"Laporan Analisis Sentimen • {datetime.now().strftime('%d-%m-%Y')}")
        canvas.restoreState()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=22,
        alignment=1,  # Center alignment
        spaceAfter=12,
        textColor=primary_color,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=16,
        alignment=1,
        spaceAfter=20,
        textColor=secondary_color
    )
    
    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=primary_color,
        spaceBefore=16,
        spaceAfter=10,
        underline=1,
        leftIndent=0
    )
    
    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=secondary_color,
        spaceBefore=12,
        spaceAfter=8,
        leftIndent=0
    )
    
    heading3_style = ParagraphStyle(
        'Heading3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=6,
        leftIndent=10
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        textColor=colors.black,
        leading=14,  # line spacing
        leftIndent=0
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=normal_style,
        leftIndent=15,
        bulletIndent=0,
        bulletText='•'
    )
    
    # Fungsi untuk membuat horizontal line
    def horizontal_line():
        line = HRFlowable(
            width="100%",
            thickness=1,
            color=secondary_color,
            spaceBefore=5,
            spaceAfter=10
        )
        elements.append(line)
    
    # ============= HALAMAN SAMPUL =============
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("LAPORAN ANALISIS SENTIMEN", title_style))
    elements.append(Paragraph("Platform X (Twitter)", subtitle_style))
    elements.append(Spacer(1, 0.5*inch))
    
    # Dua garis pembatas
    horizontal_line()
    elements.append(Spacer(1, 0.2*inch))
    
    # Judul Analisis
    elements.append(Paragraph(f"<b>Topik:</b> {analysis_context['title']}", heading1_style))
    elements.append(Spacer(1, 0.5*inch))
    
    # Metadata dan statistik utama
    date_text = f"<b>Tanggal Pembuatan:</b> {datetime.now().strftime('%d %B %Y')}"
    time_text = f"<b>Waktu:</b> {datetime.now().strftime('%H:%M')}"
    total_text = f"<b>Total Data:</b> {analysis_context['total_tweets']} tweets"
    period_text = f"<b>Periode Analisis:</b> {datetime.now().strftime('%B %Y')}"
    
    elements.append(Paragraph(date_text, normal_style))
    elements.append(Paragraph(time_text, normal_style))
    elements.append(Paragraph(total_text, normal_style))
    elements.append(Paragraph(period_text, normal_style))
    
    elements.append(Spacer(1, 0.5*inch))
    horizontal_line()
    
    # Statistik sentimen halaman depan
    pos_percent = analysis_context['positive_percent']
    neu_percent = analysis_context['neutral_percent']
    neg_percent = analysis_context['negative_percent']
    
    dominant = "Positif" if pos_percent >= max(neu_percent, neg_percent) else \
               "Netral" if neu_percent >= max(pos_percent, neg_percent) else "Negatif"
    
    sentiment_summary = f"""
    <b>Ringkasan Sentimen:</b><br/>
    • <font color="green">Positif</font>: {pos_percent}%<br/>
    • <font color="blue">Netral</font>: {neu_percent}%<br/> 
    • <font color="red">Negatif</font>: {neg_percent}%<br/><br/>
    <b>Sentimen Dominan:</b> {dominant}
    """
    
    elements.append(Paragraph(sentiment_summary, normal_style))
    elements.append(Spacer(1, 1*inch))
    
    # Watermark
    elements.append(Paragraph(
        "Dihasilkan oleh Aplikasi Analisis Sentimen X",
        ParagraphStyle(
            'Watermark',
            parent=normal_style,
            alignment=1,
            textColor=colors.gray,
            fontSize=10
        )
    ))
    
    # Page break
    elements.append(PageBreak())
    
    # ============= DAFTAR ISI =============
    elements.append(Paragraph("DAFTAR ISI", heading1_style))
    horizontal_line()
    
    toc_items = [
        "<b>1. Ringkasan Eksekutif</b>",
        "<b>2. Analisis Sentimen</b>",
        "   2.1 Distribusi Sentimen",
        "   2.2 Visualisasi Sentimen",
        "<b>3. Topik dan Hashtag Utama</b>",
        "   3.1 Topik Utama",
        "   3.2 Hashtag Populer",
        "<b>4. Sampel Tweet</b>",
        "   4.1 Tweet Positif",
        "   4.2 Tweet Netral",
        "   4.3 Tweet Negatif",
        "<b>5. Kesimpulan dan Rekomendasi</b>",
        "   5.1 Kesimpulan",
        "   5.2 Rekomendasi",
        "   5.3 Metodologi Analisis",
    ]
    
    for item in toc_items:
        elements.append(Paragraph(item, normal_style))
        elements.append(Spacer(1, 5))
    
    elements.append(Spacer(1, 0.5*inch))
    elements.append(PageBreak())
    
    # ============= RINGKASAN EKSEKUTIF =============
    elements.append(Paragraph("1. RINGKASAN EKSEKUTIF", heading1_style))
    horizontal_line()
    
    executive_summary = f"""
    Laporan ini menyajikan hasil analisis sentimen terhadap {analysis_context['total_tweets']} tweets 
    yang terkait dengan "{analysis_context['title']}" pada platform X (Twitter). Data dianalisis 
    menggunakan model pembelajaran mesin (machine learning) berbasis IndoBERT yang terlatih 
    untuk mengklasifikasikan sentimen ke dalam tiga kategori: Positif, Netral, dan Negatif.
    """
    elements.append(Paragraph(executive_summary, normal_style))
    
    elements.append(Paragraph("Temuan Utama:", heading3_style))
    
    findings_text = f"""
    • Dari total {analysis_context['total_tweets']} tweets yang dianalisis, <b>{pos_percent}%</b> memiliki 
      sentimen positif, <b>{neu_percent}%</b> netral, dan <b>{neg_percent}%</b> negatif.<br/><br/>
      
    • Sentimen <b>{dominant}</b> mendominasi percakapan tentang topik ini, 
      menunjukkan bahwa publik secara umum memiliki pandangan {'positif' if dominant == 'Positif' else
      'netral' if dominant == 'Netral' else 'negatif'} terhadap "{analysis_context['title']}".<br/><br/>
      
    • Topik-topik utama yang sering dibicarakan meliputi: {', '.join(analysis_context['top_topics'][:3])}.<br/><br/>
    
    • Hashtag populer yang sering digunakan dalam tweets meliputi: {', '.join(analysis_context['top_hashtags'][:3])}.<br/><br/>
    """
    elements.append(Paragraph(findings_text, normal_style))
    
    # Insight tambahan berdasarkan sentimen dominan
    insight_text = {
        "Positif": f"""
        <b>Insight Utama:</b> Dominasi sentimen positif ({pos_percent}%) menunjukkan adanya 
        penerimaan dan dukungan publik yang baik terhadap topik ini. Penting untuk mempertahankan 
        momentum positif ini dengan terus mengkomunikasikan aspek-aspek yang mendapatkan respons baik.<br/><br/>
        """,
        
        "Netral": f"""
        <b>Insight Utama:</b> Dominasi sentimen netral ({neu_percent}%) menunjukkan bahwa banyak 
        percakapan bersifat informatif atau faktual tanpa menunjukkan emosi kuat. Ini menandakan perlu 
        upaya lebih untuk menggeser sentimen ke arah positif dengan mengkomunikasikan nilai dan manfaat.<br/><br/>
        """,
        
        "Negatif": f"""
        <b>Insight Utama:</b> Dominasi sentimen negatif ({neg_percent}%) menunjukkan adanya 
        kekhawatiran atau ketidakpuasan publik terhadap topik ini. Penting untuk mengidentifikasi 
        sumber masalah utama dan memberikan klarifikasi serta solusi untuk memperbaiki persepsi.<br/><br/>
        """
    }
    
    elements.append(Paragraph(insight_text[dominant], normal_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(PageBreak())
    
    # ============= ANALISIS SENTIMEN =============
    elements.append(Paragraph("2. ANALISIS SENTIMEN", heading1_style))
    horizontal_line()
    
    # Distribusi Sentimen
    elements.append(Paragraph("2.1 Distribusi Sentimen", heading2_style))
    
    sentiment_table_content = f"""
    <b>Total Tweets Dianalisis:</b> {analysis_context['total_tweets']}<br/><br/>
    <font color="green"><b>Tweet Positif:</b> {analysis_context['positive_count']} ({pos_percent}%)</font><br/>
    <font color="blue"><b>Tweet Netral:</b> {analysis_context['neutral_count']} ({neu_percent}%)</font><br/>
    <font color="red"><b>Tweet Negatif:</b> {analysis_context['negative_count']} ({neg_percent}%)</font>
    """
    
    elements.append(Paragraph(sentiment_table_content, 
        ParagraphStyle('BoxedContent', 
                      parent=normal_style,
                      backColor=light_color,
                      borderColor=secondary_color,
                      borderWidth=1,
                      borderPadding=10,
                      borderRadius=5,
                      spaceBefore=10,
                      spaceAfter=10)))
    
    # Analisis Trend
    sentiment_analysis = f"""
    <b>Analisis Sentimen:</b><br/><br/>
    Berdasarkan data di atas, sentimen <b>{dominant}</b> mendominasi percakapan tentang 
    "{analysis_context['title']}" dengan persentase {pos_percent if dominant == 'Positif' else 
    neu_percent if dominant == 'Netral' else neg_percent}%. 
    """
    elements.append(Paragraph(sentiment_analysis, normal_style))
    
    # Visualisasi Sentimen
    elements.append(Paragraph("2.2 Visualisasi Sentimen", heading2_style))
    
    # Diagram Lingkaran (Pie Chart)
    elements.append(Paragraph("Distribusi Sentimen (Pie Chart):", heading3_style))
    
    # Buat pie chart
    drawing = Drawing(400, 200)
    pie = Pie()
    pie.x = 150
    pie.y = 25
    pie.width = 150
    pie.height = 150
    pie.data = [analysis_context['positive_count'], analysis_context['neutral_count'], analysis_context['negative_count']]
    pie.labels = [f'Positif ({pos_percent}%)', f'Netral ({neu_percent}%)', f'Negatif ({neg_percent}%)']
    
    # Colors
    pie.slices.strokeWidth = 0.5
    pie.slices[0].fillColor = colors.green
    pie.slices[1].fillColor = colors.blue
    pie.slices[2].fillColor = colors.red
    
    # Add legend
    from reportlab.graphics.charts.legends import Legend
    legend = Legend()
    legend.alignment = 'right'
    legend.x = 320
    legend.y = 100
    legend.colorNamePairs = [(colors.green, f'Positif ({pos_percent}%)'), 
                             (colors.blue, f'Netral ({neu_percent}%)'),
                             (colors.red, f'Negatif ({neg_percent}%)')]
    
    drawing.add(pie)
    drawing.add(legend)
    elements.append(drawing)
    
    elements.append(Paragraph("Interpretasi:", heading3_style))
    chart_interpretation = f"""
    Diagram lingkaran di atas menunjukkan dominasi sentimen {dominant.lower()} dalam percakapan tentang 
    "{analysis_context['title']}". Hal ini mengindikasikan bahwa publik secara umum {'mendukung dan memiliki pandangan positif' if dominant == 'Positif' else 'bersikap netral dan objektif' if dominant == 'Netral' 
    else 'memiliki kekhawatiran dan kritik'} terhadap topik ini.
    """
    elements.append(Paragraph(chart_interpretation, normal_style))
    elements.append(PageBreak())
    
    # ============= TOPIK DAN HASHTAG UTAMA =============
    elements.append(Paragraph("3. TOPIK DAN HASHTAG UTAMA", heading1_style))
    horizontal_line()
    
    # Topik Utama
    elements.append(Paragraph("3.1 Topik Utama", heading2_style))
    
    topic_intro = f"""
    Berikut adalah topik-topik utama yang paling sering muncul dalam tweets terkait 
    "{analysis_context['title']}":
    """
    elements.append(Paragraph(topic_intro, normal_style))
    
    if 'top_topics' in analysis_context and analysis_context['top_topics']:
        # Format topic list with bullet points
        topics_list = ""
        for i, topic in enumerate(analysis_context['top_topics'][:8], 1):
            topics_list += f"<b>{i}.</b> {topic}<br/>"
        
        # Display topics in a box
        elements.append(Paragraph(topics_list, 
            ParagraphStyle('BoxedTopics', 
                          parent=normal_style,
                          backColor=light_color,
                          borderColor=secondary_color,
                          borderWidth=1,
                          borderPadding=10,
                          spaceBefore=10,
                          spaceAfter=10)))
    else:
        elements.append(Paragraph("Data topik utama tidak tersedia.", normal_style))
    
    # Topic Analysis
    topic_analysis = f"""
    <b>Analisis Topik:</b><br/><br/>
    Topik-topik utama yang dibicarakan mencerminkan fokus perhatian publik terkait 
    "{analysis_context['title']}". Dengan memahami topik-topik ini, kita dapat menyesuaikan
    strategi komunikasi untuk lebih efektif menjangkau audiens target.
    """
    elements.append(Paragraph(topic_analysis, normal_style))
    
    # Hashtag Populer
    elements.append(Paragraph("3.2 Hashtag Populer", heading2_style))
    
    hashtag_intro = f"""
    Berikut adalah hashtag yang paling sering digunakan dalam percakapan terkait 
    "{analysis_context['title']}":
    """
    elements.append(Paragraph(hashtag_intro, normal_style))
    
    if 'top_hashtags' in analysis_context and analysis_context['top_hashtags']:
        # Format hashtag list with bullet points
        hashtags_list = ""
        for i, hashtag in enumerate(analysis_context['top_hashtags'][:8], 1):
            hashtag_tag = hashtag if isinstance(hashtag, str) else hashtag.get('tag', 'Unknown')
            hashtags_list += f"<b>{i}.</b> {hashtag_tag}<br/>"
        
        # Display hashtags in a box
        elements.append(Paragraph(hashtags_list, 
            ParagraphStyle('BoxedHashtags', 
                          parent=normal_style,
                          backColor=light_color,
                          borderColor=secondary_color,
                          borderWidth=1,
                          borderPadding=10,
                          spaceBefore=10,
                          spaceAfter=10)))
    else:
        elements.append(Paragraph("Data hashtag populer tidak tersedia.", normal_style))
    
    # Hashtag Analysis
    hashtag_analysis = f"""
    <b>Analisis Hashtag:</b><br/><br/>
    Hashtag yang populer memberikan gambaran tentang bagaimana percakapan dikategorikan dan
    diorganisir di media sosial. Hashtag ini juga dapat digunakan untuk melacak percakapan
    terkait dan meningkatkan jangkauan komunikasi.
    """
    elements.append(Paragraph(hashtag_analysis, normal_style))
    elements.append(PageBreak())
    
    # ============= SAMPEL TWEET =============
    elements.append(Paragraph("4. SAMPEL TWEET", heading1_style))
    horizontal_line()
    
    # Helper function untuk memotong teks yang terlalu panjang
    def truncate_text(text, max_length=100):
        if text and len(text) > max_length:
            return text[:max_length] + "..."
        return text if text else ""
    
    # Fungsi untuk menambahkan sampel tweet dengan styling yang lebih baik
    def add_tweet_samples(section_num, category, sentiment_type, color):
        elements.append(Paragraph(f"4.{section_num} Tweet {category}", heading2_style))
        elements.append(Paragraph(f"Berikut adalah sampel tweet dengan sentimen {category.lower()}:", normal_style))
        
        # Filter tweets by sentiment
        tweets = analysis_df[analysis_df['predicted_sentiment'] == sentiment_type].head(3)
        
        if len(tweets) > 0:
            for i, (_, tweet) in enumerate(tweets.iterrows(), 1):
                username = tweet.get('username', 'user')
                content = truncate_text(tweet.get('content', ''), 150)
                confidence = f"{tweet.get('confidence', 0):.1f}%"
                date = tweet.get('date', datetime.now().strftime('%d %b %Y'))
                
                # Format tweet
                tweet_text = f"""
                <font color="blue"><b>@{username}</b></font> • {date}<br/>
                {content}<br/>
                <font color="gray"><i>Confidence: {confidence}</i></font>
                """
                
                # Custom tweet style with background color
                tweet_style = ParagraphStyle(
                    f'Tweet{sentiment_type}{i}',
                    parent=normal_style,
                    backColor=light_color,
                    borderColor=color,
                    borderWidth=1,
                    borderPadding=8,
                    spaceBefore=10,
                    spaceAfter=10
                )
                
                elements.append(Paragraph(tweet_text, tweet_style))
        else:
            elements.append(Paragraph(f"Tidak ada tweet {category.lower()} yang tersedia.", normal_style))
        
        # Add analysis for each sentiment type
        analysis_text = {
            "Positif": """
            <b>Karakteristik Tweet Positif:</b><br/>
            Tweet positif umumnya mengandung ekspresi dukungan, apresiasi, atau optimisme. 
            Kata-kata yang sering muncul antara lain: "bagus", "sukses", "terima kasih", 
            "mendukung", "setuju", dll.<br/><br/>
            """,
            
            "Netral": """
            <b>Karakteristik Tweet Netral:</b><br/>
            Tweet netral umumnya bersifat informatif, faktual, atau berupa pertanyaan. 
            Tweet ini tidak menunjukkan emosi atau pendapat yang kuat, baik positif maupun negatif.<br/><br/>
            """,
            
            "Negatif": """
            <b>Karakteristik Tweet Negatif:</b><br/>
            Tweet negatif umumnya mengandung ekspresi kritik, kekecewaan, atau kekhawatiran. 
            Kata-kata yang sering muncul antara lain: "kecewa", "masalah", "buruk", "gagal", 
            "tidak setuju", dll.<br/><br/>
            """
        }
        
        elements.append(Paragraph(analysis_text[sentiment_type], normal_style))
        elements.append(Spacer(1, 0.2*inch))
    
    # Tambahkan sampel tweet untuk setiap sentimen
    add_tweet_samples(1, "Positif", "Positif", colors.green)
    add_tweet_samples(2, "Netral", "Netral", colors.blue)
    add_tweet_samples(3, "Negatif", "Negatif", colors.red)
    
    elements.append(PageBreak())
    
    # ============= KESIMPULAN DAN REKOMENDASI =============
    elements.append(Paragraph("5. KESIMPULAN DAN REKOMENDASI", heading1_style))
    horizontal_line()
    
    # Kesimpulan
    elements.append(Paragraph("5.1 Kesimpulan", heading2_style))
    
    conclusion_text = f"""
    Berdasarkan analisis sentimen terhadap {analysis_context['total_tweets']} tweets terkait 
    "{analysis_context['title']}", dapat disimpulkan bahwa:<br/><br/>
    
    1. Sentimen publik secara keseluruhan cenderung {'positif' if dominant == 'Positif' else
       'netral' if dominant == 'Netral' else 'negatif'} dengan persentase {pos_percent if dominant == 'Positif' 
       else neu_percent if dominant == 'Netral' else neg_percent}%.<br/><br/>
    
    2. Topik-topik utama yang dibicarakan adalah seputar {', '.join(analysis_context['top_topics'][:3])}.<br/><br/>
    
    3. Hashtag yang paling sering digunakan adalah {', '.join(analysis_context['top_hashtags'][:3])}.<br/><br/>
    
    4. Percakapan di media sosial X menunjukkan {'tingkat dukungan dan antusiasme yang baik' 
       if dominant == 'Positif' else 'sikap yang cenderung netral dan informatif' 
       if dominant == 'Netral' else 'adanya kekhawatiran dan kritik'} terhadap topik ini.<br/><br/>
    """
    
    elements.append(Paragraph(conclusion_text, normal_style))
    
    # Rekomendasi berdasarkan sentimen dominan
    elements.append(Paragraph("5.2 Rekomendasi", heading2_style))
    
    recommendation_text = {
        "Positif": f"""
        Berdasarkan dominasi sentimen positif, berikut adalah rekomendasi untuk mempertahankan
        dan meningkatkan persepsi positif publik:<br/><br/>
        
        1. <b>Pertahankan Momentum Positif</b> - Teruskan komunikasi aspek-aspek yang mendapatkan
           respon positif dari publik.<br/><br/>
        
        2. <b>Manfaatkan Pendukung</b> - Identifikasi dan libatkan pendukung aktif untuk
           memperluas jangkauan pesan positif.<br/><br/>
        
        3. <b>Gunakan Hashtag Populer</b> - Manfaatkan hashtag {', '.join(analysis_context['top_hashtags'][:2])}
           untuk meningkatkan visibilitas pesan.<br/><br/>
        
        4. <b>Eksplorasi Topik Potensial</b> - Kembangkan konten seputar {', '.join(analysis_context['top_topics'][:2])}
           yang mendapat respons positif.<br/><br/>
        
        5. <b>Pantau Secara Berkala</b> - Lakukan analisis sentimen secara berkala untuk
           mendeteksi perubahan tren dan menyesuaikan strategi.<br/><br/>
        """,
        
        "Netral": f"""
        Berdasarkan dominasi sentimen netral, berikut adalah rekomendasi untuk meningkatkan
        engagement dan menggeser sentimen ke arah yang lebih positif:<br/><br/>
        
        1. <b>Tingkatkan Komunikasi Nilai</b> - Perkuat pesan tentang manfaat dan nilai positif
           untuk menggeser sentimen dari netral ke positif.<br/><br/>
        
        2. <b>Edukasi Publik</b> - Lakukan edukasi yang lebih intensif tentang aspek-aspek
           penting yang mungkin belum dipahami sepenuhnya.<br/><br/>
        
        3. <b>Ciptakan Konten Engaging</b> - Kembangkan konten yang lebih menarik dan memicu
           respons emosional positif.<br/><br/>
        
        4. <b>Gunakan Influencer</b> - Libatkan influencer untuk memperkuat pesan dan menciptakan
           sentimen yang lebih positif.<br/><br/>
        
        5. <b>Pantau Tren Pergeseran</b> - Perhatikan percakapan netral yang berpotensi
           bergeser ke arah positif atau negatif.<br/><br/>
        """,
        
        "Negatif": f"""
        Berdasarkan dominasi sentimen negatif, berikut adalah rekomendasi untuk memperbaiki
        persepsi publik:<br/><br/>
        
        1. <b>Identifikasi Masalah Utama</b> - Lakukan analisis mendalam terhadap sumber
           kekhawatiran dan kritik utama.<br/><br/>
        
        2. <b>Berikan Klarifikasi</b> - Komunikasikan klarifikasi untuk isu-isu yang sering
           mendapat kritik.<br/><br/>
        
        3. <b>Tunjukkan Langkah Perbaikan</b> - Informasikan tentang langkah-langkah konkret
           yang sedang atau akan dilakukan untuk mengatasi masalah.<br/><br/>
        
        4. <b>Engagement Aktif</b> - Tingkatkan keterlibatan dengan audiens kritis untuk
           menunjukkan keterbukaan terhadap umpan balik.<br/><br/>
        
        5. <b>Pemantauan Intensif</b> - Lakukan pemantauan lebih sering untuk melihat
           perubahan sentimen setelah implementasi rekomendasi.<br/><br/>
        """
    }
    
    elements.append(Paragraph(recommendation_text[dominant], normal_style))
    
    # Metodologi
    elements.append(Paragraph("5.3 Metodologi Analisis", heading2_style))
    
    methodology_text = f"""
    Analisis sentimen dalam laporan ini menggunakan model machine learning berbasis IndoBERT
    yang dilatih khusus untuk mengklasifikasikan teks Bahasa Indonesia ke dalam tiga kategori
    sentimen: Positif, Netral, dan Negatif.<br/><br/>
    
    Model ini memiliki akurasi sekitar 85% berdasarkan validasi pada dataset pengujian. Persentase
    kepercayaan (confidence score) ditampilkan untuk setiap sampel tweet sebagai indikasi
    tingkat keyakinan model terhadap prediksi yang dihasilkan.<br/><br/>
    
    Data tweets diambil dari platform X (Twitter) menggunakan Scraping data dengan filter
    berdasarkan kata kunci yang relevan dengan topik "{analysis_context['title']}".<br/><br/>
    """
    
    elements.append(Paragraph(methodology_text, normal_style))
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(
        f"Laporan ini dibuat secara otomatis oleh Aplikasi Analisis Sentimen X • {datetime.now().strftime('%d %B %Y')}",
        ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            alignment=1,  # Center
            textColor=colors.gray
        )
    ))
    
    # Build PDF with page numbers
    doc.build(elements, onFirstPage=add_page_number, onLaterPages=add_page_number)
    buffer.seek(0)
    
    # Return file
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Laporan_Analisis_Sentimen_{analysis_context['title'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    # Cek keberadaan model pada startup
    if not os.path.exists(MODEL_PATH):
        print("=" * 80)
        print(f"PERINGATAN: Model {MODEL_PATH} tidak ditemukan!")
        print("Pastikan file model indobert_sentiment_best.pt sudah ada di folder models/")
        print("Aplikasi akan tetap berjalan, tetapi analisis sentimen tidak akan berfungsi sampai model tersedia.")
        print("=" * 80)
    
    app.run(debug=True)