from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from flask_cors import CORS

from backend.config import Config
# Import the auth blueprint (we’ll create backend/auth.py next)
from backend.auth import auth_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS
    CORS(app)

    # Setup JWT
    JWTManager(app)

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix="/api/auth")

    # Health check route
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "API running"}), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)