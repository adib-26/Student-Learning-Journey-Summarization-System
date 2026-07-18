import os

from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager
from flask_cors import CORS

from backend.config import Config
from backend.api_routes import api_bp
from backend.api_storage import ApiDatabase
from backend.api_utils import ApiError, error_response


def create_app(config_override=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if config_override:
        app.config.update(config_override)

    CORS(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})

    database = ApiDatabase(app.config["API_DATABASE_PATH"])
    database.initialize()
    app.extensions["api_db"] = database

    jwt = JWTManager(app)
    app.register_blueprint(api_bp)

    @app.errorhandler(ApiError)
    def handle_api_error(error):
        return error_response(error.code, error.message, error.status_code, error.details)

    @app.errorhandler(413)
    def handle_request_too_large(_error):
        return error_response("PAYLOAD_TOO_LARGE", "The request is too large.", 413)

    @app.errorhandler(404)
    def handle_not_found(_error):
        if request.path.startswith("/api/"):
            return error_response("NOT_FOUND", "The requested resource was not found.", 404)
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(_error):
        return error_response("METHOD_NOT_ALLOWED", "The HTTP method is not allowed for this resource.", 405)

    @jwt.unauthorized_loader
    def handle_missing_token(message):
        return error_response("UNAUTHORIZED", message, 401)

    @jwt.invalid_token_loader
    def handle_invalid_token(message):
        return error_response("UNAUTHORIZED", message, 401)

    @jwt.expired_token_loader
    def handle_expired_token(_header, _payload):
        return error_response("TOKEN_EXPIRED", "The access token has expired.", 401)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "5000")),
    )
