"""
JWT Authentication Module
=========================
Generates, validates, and decodes Access & Refresh Tokens using PyJWT.
Supports Authorization Bearer headers and HTTPOnly Cookies.
"""

import os
import jwt
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Any, Optional, Tuple, List
from flask import request, jsonify, current_app, g


def get_jwt_secret() -> str:
    """Retrieve the application secret key for JWT signing."""
    if current_app:
        return current_app.config.get('SECRET_KEY') or 'ai-nids-super-secret-jwt-key'
    return os.environ.get('SECRET_KEY', 'ai-nids-super-secret-jwt-key')


def generate_access_token(user, expires_in_minutes: int = 15) -> str:
    """
    Generate short-lived JWT access_token (15 minutes).
    
    Payload contains:
    - sub (user id)
    - username
    - email
    - role
    - type: 'access'
    - exp, iat
    """
    secret = get_jwt_secret()
    now = datetime.utcnow()
    payload = {
        'sub': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role,
        'type': 'access',
        'iat': now,
        'exp': now + timedelta(minutes=expires_in_minutes)
    }
    return jwt.encode(payload, secret, algorithm='HS256')


def generate_refresh_token(user, expires_in_days: int = 7) -> str:
    """
    Generate long-lived JWT refresh_token (7 days).
    
    Payload contains:
    - sub (user id)
    - username
    - type: 'refresh'
    - exp, iat
    """
    secret = get_jwt_secret()
    now = datetime.utcnow()
    payload = {
        'sub': user.id,
        'username': user.username,
        'type': 'refresh',
        'iat': now,
        'exp': now + timedelta(days=expires_in_days)
    }
    return jwt.encode(payload, secret, algorithm='HS256')


def generate_token_pair(user) -> Dict[str, Any]:
    """
    Generate both access_token and refresh_token for a user.
    """
    access_tok = generate_access_token(user, expires_in_minutes=15)
    refresh_tok = generate_refresh_token(user, expires_in_days=7)
    return {
        'access_token': access_tok,
        'refresh_token': refresh_tok,
        'token_type': 'Bearer',
        'expires_in': 15 * 60,  # 900 seconds
        'user': user.to_dict() if hasattr(user, 'to_dict') else {'id': user.id, 'username': user.username}
    }


def decode_jwt_token(token: str, expected_type: str = 'access') -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Decode and validate a JWT token.
    
    Returns:
        (is_valid: bool, payload: dict | None, error_message: str | None)
    """
    secret = get_jwt_secret()
    try:
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        token_type = payload.get('type')
        if expected_type and token_type != expected_type:
            return False, None, f"Invalid token type: expected '{expected_type}', got '{token_type}'"
        return True, payload, None
    except jwt.ExpiredSignatureError:
        return False, None, "Token has expired. Please refresh or sign in again."
    except jwt.InvalidTokenError as e:
        return False, None, f"Invalid token: {str(e)}"
    except Exception as e:
        return False, None, f"Token validation error: {str(e)}"


def extract_token_from_request(cookie_name: str = 'access_token') -> Optional[str]:
    """
    Extract token from 'Authorization: Bearer <token>' header or fallback to cookie.
    """
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header.split(' ', 1)[1].strip()
    
    # Check custom X-Access-Token header
    if request.headers.get('X-Access-Token'):
        return request.headers.get('X-Access-Token').strip()
        
    # Check cookie
    if request.cookies and cookie_name in request.cookies:
        return request.cookies.get(cookie_name)
        
    return None


def jwt_required(roles: Optional[List[str]] = None):
    """
    Decorator for endpoints requiring a valid access_token.
    Optionally enforces RBAC roles (e.g. ['admin', 'analyst']).
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from app.models.database import User
            
            token = extract_token_from_request('access_token')
            if not token:
                return jsonify({
                    'status': 'error',
                    'error': 'Missing access_token',
                    'message': 'Authorization header (Bearer token) or access_token cookie required.'
                }), 401
            
            is_valid, payload, error_msg = decode_jwt_token(token, expected_type='access')
            if not is_valid:
                return jsonify({
                    'status': 'error',
                    'error': 'Invalid access_token',
                    'message': error_msg
                }), 401
            
            user_id = payload.get('sub')
            user = User.query.get(user_id)
            if not user or not user.is_active:
                return jsonify({
                    'status': 'error',
                    'error': 'User not found or inactive'
                }), 401
                
            # Check role permission if specified
            if roles and user.role not in roles:
                return jsonify({
                    'status': 'error',
                    'error': 'Forbidden',
                    'message': f"Role '{user.role}' lacks required permissions: {roles}"
                }), 403
                
            # Attach authenticated user to request context
            g.jwt_user = user
            g.jwt_payload = payload
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def jwt_refresh_required(f):
    """
    Decorator for the /refresh endpoint requiring a valid refresh_token.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from app.models.database import User
        
        token = extract_token_from_request('refresh_token')
        if not token:
            # Also check body for refresh_token
            if request.is_json:
                token = request.json.get('refresh_token')
            elif request.form:
                token = request.form.get('refresh_token')
                
        if not token:
            return jsonify({
                'status': 'error',
                'error': 'Missing refresh_token',
                'message': 'refresh_token must be provided in header, body, or cookie.'
            }), 401
            
        is_valid, payload, error_msg = decode_jwt_token(token, expected_type='refresh')
        if not is_valid:
            return jsonify({
                'status': 'error',
                'error': 'Invalid refresh_token',
                'message': error_msg
            }), 401
            
        user_id = payload.get('sub')
        user = User.query.get(user_id)
        if not user or not user.is_active:
            return jsonify({
                'status': 'error',
                'error': 'User not found or inactive'
            }), 401
            
        g.jwt_user = user
        g.jwt_payload = payload
        return f(*args, **kwargs)
    return decorated_function
