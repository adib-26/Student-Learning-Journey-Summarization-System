import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "fyp2-secret-key")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fyp2-jwt-secret")