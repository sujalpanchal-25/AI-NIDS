"""
Authentication Routes
=====================
User authentication, registration, and session management.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, Response
from flask_login import login_user, logout_user, login_required, current_user
from urllib.parse import urlparse as url_parse
import json
from datetime import datetime

from app import db, login_manager
from app.models.database import User, Alert, APIKey
from app.routes.forms import LoginForm, RegistrationForm, ChangePasswordForm, VerifyEmailForm

auth_bp = Blueprint('auth', __name__)


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login."""
    return User.query.get(int(user_id))


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
    
    form = LoginForm()
    
    if form.validate_on_submit():
        username_clean = form.username.data.strip()
        user = User.query.filter_by(username=username_clean).first()
        
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('auth.login'))
        
        # Check if email is verified
        if not user.is_verified:
            otp = user.generate_otp()
            db.session.commit()
            
            try:
                from utils.email_service import send_verification_otp_email_async
                send_verification_otp_email_async(
                    user_email=user.email,
                    username=user.username,
                    otp_code=otp
                )
            except Exception as e_otp:
                from flask import current_app
                current_app.logger.warning(f"Failed to send verification OTP on login: {e_otp}")
                
            flash('Please verify your email address to activate your account. A 6-digit verification code has been sent to your email.', 'warning')
            return redirect(url_for('auth.verify_email', email=user.email))
        
        if not user.is_active:
            flash('Your account has been deactivated. Please contact an administrator.', 'warning')
            return redirect(url_for('auth.login'))
        
        from flask import session
        from datetime import timedelta
        
        remember_flag = bool(form.remember_me.data)
        session.permanent = remember_flag
        
        # Reset dataset selection and set clean session start for fresh 0 metrics
        session.pop('selected_dataset', None)
        session.pop('selected_dataset_name', None)
        session['session_start_time'] = datetime.utcnow().isoformat()
        
        login_user(
            user, 
            remember=remember_flag, 
            duration=timedelta(days=1) if remember_flag else None
        )
        user.last_login = db.func.now()
        db.session.commit()
        
        flash(f'Welcome back, {user.username}!', 'success')
        
        # Redirect to next page or dashboard
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('dashboard.dashboard')
        
        return redirect(next_page)
    
    return render_template('login.html', form=form)


@auth_bp.route('/logout')
@login_required
def logout():
    """User logout."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration with Brevo email OTP validation."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
    
    form = RegistrationForm()
    
    if form.validate_on_submit():
        user = User(
            username=form.username.data.strip(),
            email=form.email.data.strip().lower(),
            role='analyst',
            is_active=False,     # Inactive until email is verified
            is_verified=False
        )
        user.set_password(form.password.data)
        otp = user.generate_otp()
        
        db.session.add(user)
        db.session.commit()
        
        # Send 6-digit OTP verification email via Brevo API
        try:
            from utils.email_service import send_verification_otp_email_async
            send_verification_otp_email_async(
                user_email=user.email,
                username=user.username,
                otp_code=otp
            )
        except Exception as e_mail:
            from flask import current_app
            current_app.logger.warning(f"Failed to dispatch OTP email: {e_mail}")
        
        flash(f'Registration initiated! A 6-digit verification code has been sent to {user.email}. Enter it below to activate your account.', 'info')
        return redirect(url_for('auth.verify_email', email=user.email))
    
    return render_template('register.html', form=form)


@auth_bp.route('/verify-email', methods=['GET', 'POST'])
def verify_email():
    """Verify 6-digit OTP email confirmation."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
    
    target_email = request.args.get('email', '') or request.form.get('email', '')
    form = VerifyEmailForm()
    
    if request.method == 'GET' and target_email:
        form.email.data = target_email
        
    if form.validate_on_submit():
        email_clean = form.email.data.strip().lower()
        otp_code = form.otp.data.strip()
        
        user = User.query.filter_by(email=email_clean).first()
        
        if not user:
            flash('No account found with this email address.', 'danger')
            return render_template('verify_email.html', form=form, target_email=email_clean)
        
        if user.is_verified:
            flash('Your email is already verified! Please sign in.', 'info')
            return redirect(url_for('auth.login'))
        
        success, message = user.verify_otp(otp_code)
        
        if not success:
            flash(message, 'danger')
            return render_template('verify_email.html', form=form, target_email=email_clean)
        
        user.last_login = db.func.now()
        db.session.commit()
        
        from flask import session
        session.pop('selected_dataset', None)
        session.pop('selected_dataset_name', None)
        session['session_start_time'] = datetime.utcnow().isoformat()
        
        # Log user in
        login_user(user)
        
        # Send confirmation / welcome email via Brevo
        try:
            from utils.email_service import send_registration_confirmation_async
            send_registration_confirmation_async(
                user_email=user.email,
                username=user.username,
                role=user.role
            )
        except Exception as e_conf:
            from flask import current_app
            current_app.logger.warning(f"Failed to dispatch welcome email: {e_conf}")
            
        flash(f'Email verified successfully! Welcome to AI-NIDS, {user.username}!', 'success')
        return redirect(url_for('dashboard.dashboard'))
        
    return render_template('verify_email.html', form=form, target_email=target_email)


@auth_bp.route('/resend-otp', methods=['POST'])
def resend_otp():
    """Resend 6-digit OTP code to user email via Brevo API."""
    email = request.form.get('email', '').strip().lower()
    
    if not email:
        flash('Email address is required to resend verification code.', 'warning')
        return redirect(url_for('auth.verify_email'))
    
    user = User.query.filter_by(email=email).first()
    
    if not user:
        flash('No account found with this email address.', 'danger')
        return redirect(url_for('auth.register'))
    
    if user.is_verified:
        flash('Your account is already verified! Please sign in.', 'info')
        return redirect(url_for('auth.login'))
    
    otp = user.generate_otp()
    db.session.commit()
    
    try:
        from utils.email_service import send_verification_otp_email_async
        send_verification_otp_email_async(
            user_email=user.email,
            username=user.username,
            otp_code=otp
        )
    except Exception as e_resend:
        from flask import current_app
        current_app.logger.warning(f"Failed to resend OTP email: {e_resend}")
        
    flash(f'A new 6-digit verification code has been sent to {user.email}.', 'success')
    return redirect(url_for('auth.verify_email', email=user.email))



@auth_bp.route('/profile')
@login_required
def profile():
    """User profile page."""
    from app.models.database import APIKey
    form = ChangePasswordForm()
    api_keys = APIKey.query.filter_by(user_id=current_user.id).order_by(APIKey.created_at.desc()).all()
    return render_template('profile.html', user=current_user, form=form, api_keys=api_keys)


@auth_bp.route('/profile/generate-api-key', methods=['POST'])
@login_required
def generate_api_key():
    """Generate a new API key for the current user."""
    from app.models.database import APIKey
    import secrets
    
    key_name = request.form.get('key_name', 'Default Key')
    
    # Generate a secure API key
    api_key = secrets.token_urlsafe(32)
    
    new_key = APIKey(
        key=api_key,
        name=key_name,
        user_id=current_user.id,
        is_active=True
    )
    
    db.session.add(new_key)
    db.session.commit()
    
    flash(f'API Key generated successfully! Key: {api_key}', 'success')
    return redirect(url_for('auth.profile'))


@auth_bp.route('/profile/revoke-api-key/<int:key_id>', methods=['POST'])
@login_required
def revoke_api_key(key_id):
    """Revoke an API key."""
    from app.models.database import APIKey
    
    key = APIKey.query.filter_by(id=key_id, user_id=current_user.id).first()
    if key:
        db.session.delete(key)
        db.session.commit()
        flash('API Key revoked successfully.', 'success')
    else:
        flash('API Key not found.', 'danger')
    
    return redirect(url_for('auth.profile'))


@auth_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change user password."""
    form = ChangePasswordForm()
    
    if form.validate_on_submit():
        if not current_user.check_password(form.current_password.data):
            flash('Current password is incorrect', 'danger')
            return redirect(url_for('auth.change_password'))
        
        current_user.set_password(form.new_password.data)
        db.session.commit()
        
        flash('Password changed successfully!', 'success')
        return redirect(url_for('auth.profile'))
    
    return render_template('change_password.html', form=form)


@auth_bp.route('/users')
@login_required
def user_list():
    """List all users (admin only)."""
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard.dashboard'))
    
    users = User.query.all()
    return render_template('users.html', users=users)


@auth_bp.route('/users/<int:user_id>/toggle-active', methods=['POST'])
@login_required
def toggle_user_active(user_id):
    """Toggle user active status (admin only)."""
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard.dashboard'))
    
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('You cannot deactivate yourself.', 'warning')
        return redirect(url_for('auth.user_list'))
    
    user.is_active = not user.is_active
    db.session.commit()
    
    status = 'activated' if user.is_active else 'deactivated'
    flash(f'User {user.username} has been {status}.', 'success')
    
    return redirect(url_for('auth.user_list'))


@auth_bp.route('/users/<int:user_id>/change-role', methods=['POST'])
@login_required
def change_user_role(user_id):
    """Change user role (admin only)."""
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard.dashboard'))
    
    user = User.query.get_or_404(user_id)
    new_role = request.form.get('role')
    
    if new_role not in ['admin', 'analyst', 'viewer']:
        flash('Invalid role.', 'danger')
        return redirect(url_for('auth.user_list'))
    
    if user.id == current_user.id and new_role != 'admin':
        flash('You cannot demote yourself.', 'warning')
        return redirect(url_for('auth.user_list'))
    
    user.role = new_role
    db.session.commit()
    
    flash(f'User {user.username} role changed to {new_role}.', 'success')
    return redirect(url_for('auth.user_list'))


@auth_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    """Delete user account (admin only)."""
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard.dashboard'))
    
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'warning')
        return redirect(url_for('auth.user_list'))
    
    # Delete associated API keys first
    APIKey.query.filter_by(user_id=user.id).delete()
    
    username = user.username
    db.session.delete(user)
    db.session.commit()
    
    flash(f'User {username} has been deleted successfully.', 'success')
    return redirect(url_for('auth.user_list'))



@auth_bp.route('/export-my-data')
@login_required
def export_my_data():
    """Export all user data as JSON."""
    # Collect user profile data
    user_data = {
        'export_date': datetime.utcnow().isoformat(),
        'user': {
            'username': current_user.username,
            'email': current_user.email,
            'role': current_user.role,
            'created_at': current_user.created_at.isoformat() if current_user.created_at else None,
            'last_login': current_user.last_login.isoformat() if current_user.last_login else None,
        },
        'api_keys': [],
        'alerts': []
    }
    
    # Get user's API keys (without exposing full keys)
    api_keys = APIKey.query.filter_by(user_id=current_user.id).all()
    for key in api_keys:
        user_data['api_keys'].append({
            'name': key.name,
            'created_at': key.created_at.isoformat() if key.created_at else None,
            'last_used': key.last_used.isoformat() if key.last_used else None,
            'expires_at': key.expires_at.isoformat() if key.expires_at else None,
            'is_active': key.is_active,
            'key_preview': f"{key.key[:8]}...{key.key[-4:]}" if key.key else None
        })
    
    # Get recent alerts (last 100)
    alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(100).all()
    for alert in alerts:
        user_data['alerts'].append({
            'id': alert.id,
            'attack_type': alert.attack_type,
            'severity': alert.severity,
            'source_ip': alert.source_ip,
            'destination_ip': alert.destination_ip,
            'timestamp': alert.timestamp.isoformat() if alert.timestamp else None,
            'acknowledged': alert.acknowledged,
            'resolved': alert.resolved,
            'description': alert.description
        })
    
    # Create downloadable JSON response
    response = Response(
        json.dumps(user_data, indent=2),
        mimetype='application/json',
        headers={
            'Content-Disposition': f'attachment; filename=ai-nids-export-{current_user.username}-{datetime.utcnow().strftime("%Y%m%d")}.json'
        }
    )
    
    return response


@auth_bp.route('/settings')
@login_required
def settings():
    """System settings page (admin only)."""
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard.dashboard'))
    
    return render_template('settings.html')
