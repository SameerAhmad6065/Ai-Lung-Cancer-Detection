from flask import Flask, request, jsonify, render_template, redirect, session
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import mysql.connector

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key_here'

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # your password
    database="lung_classifier_db"
)
cursor = db.cursor(dictionary=True)

# Load AI model
model = torch.jit.load('lung_cancer_classifier.pt', map_location='cpu')
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['adenocarcinoma', 'invalid', 'normal', 'squamus_cell_carcinoma']

# ------------------ ROUTES ------------------

@app.route('/')
def home():
    if 'user' in session:
        return render_template('index.html')
    return redirect('/login')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        password = request.form['password']  # Store as plain text

        try:
            cursor.execute("INSERT INTO users (fullname, email, password) VALUES (%s, %s, %s)",
                           (fullname, email, password))
            db.commit()
            return redirect('/login')
        except mysql.connector.IntegrityError:
            return "User already exists"

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password_input = request.form['password']

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            print("User found:", user)
            print("Entered password:", password_input)
            print("Stored password:", user['password'])

            if user['password'] == password_input:
                session['user'] = user['email']
                return redirect('/')
            else:
                return "Invalid credentials (password mismatch)"
        else:
            return "Invalid credentials (user not found)"

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

@app.route('/hospitals')
def hospitals():
    if 'user' not in session:
        return redirect('/login')
    return render_template('hospitals.html')

@app.route('/suggestions')
def suggestions():
    if 'user' not in session:
        return redirect('/login')
    return render_template('suggestions.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized access'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences = {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}

        return jsonify({
            'prediction': class_names[torch.argmax(probs)],
            'confidences': confidences
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ------------------ MAIN ------------------

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
