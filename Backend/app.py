from flask import Flask
from flask_cors import CORS
from controllers.query_controller import query_bp  # Import query blueprint
from controllers.upload_controller import upload_bp  # Import upload blueprint
import os

app = Flask(__name__)

# Enable CORS globally for the entire application
CORS(app)

# Register the query blueprint
app.register_blueprint(query_bp)

# Register the upload blueprint
app.register_blueprint(upload_bp)

# Set the secret key for session management (if needed)
app.secret_key = os.urandom(24)

@app.route("/")
def home():
    return "Welcome to the RAG Bot API"

if __name__ == "__main__":
    app.run(debug=True)
