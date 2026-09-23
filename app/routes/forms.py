"""
WTForms for Authentication
==========================
Form definitions for login, registration, etc.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError, Regexp

from app.models.database import User


class LoginForm(FlaskForm):
    """User login form with validations."""
    username = StringField('Username', validators=[
        DataRequired(message='Username is required.'),
        Length(min=3, max=64, message='Username must be between 3 and 64 characters.')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message='Password is required.')
    ])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class RegistrationForm(FlaskForm):
    """User registration form with comprehensive security validations."""
    username = StringField('Username', validators=[
        DataRequired(message='Username is required.'),
        Length(min=3, max=64, message='Username must be between 3 and 64 characters.'),
        Regexp(r'^[a-zA-Z0-9_-]+$', message='Username can only contain letters, numbers, underscores, and hyphens.')
    ])
    email = StringField('Email', validators=[
        DataRequired(message='Email address is required.'),
        Regexp(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', message='Please enter a valid email address.'),
        Length(max=120, message='Email address cannot exceed 120 characters.')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message='Password is required.'),
        Length(min=8, message='Password must be at least 8 characters long.')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(message='Please confirm your password.'),
        EqualTo('password', message='Passwords must match.')
    ])
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        """Check if username is already taken."""
        user = User.query.filter_by(username=username.data.strip()).first()
        if user is not None:
            raise ValidationError('Username is already taken. Please choose a different one.')
    
    def validate_email(self, email):
        """Check if email is already registered."""
        user = User.query.filter_by(email=email.data.strip().lower()).first()
        if user is not None:
            raise ValidationError('Email is already registered. Please use a different one or sign in.')

    def validate_password(self, password):
        """Ensure password meets basic complexity requirements."""
        val = password.data
        if not any(c.isupper() for c in val):
            raise ValidationError('Password must contain at least one uppercase letter (A-Z).')
        if not any(c.islower() for c in val):
            raise ValidationError('Password must contain at least one lowercase letter (a-z).')
        if not any(c.isdigit() or not c.isalnum() for c in val):
            raise ValidationError('Password must contain at least one number or special character.')



class ChangePasswordForm(FlaskForm):
    """Change password form."""
    current_password = PasswordField('Current Password', validators=[
        DataRequired()
    ])
    new_password = PasswordField('New Password', validators=[
        DataRequired(),
        Length(min=8, message='Password must be at least 8 characters')
    ])
    confirm_password = PasswordField('Confirm New Password', validators=[
        DataRequired(),
        EqualTo('new_password', message='Passwords must match')
    ])
    submit = SubmitField('Change Password')


class UserEditForm(FlaskForm):
    """Edit user form (admin)."""
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=64)
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Regexp(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', message='Please enter a valid email address.'),
        Length(max=120)
    ])
    role = SelectField('Role', choices=[
        ('viewer', 'Viewer'),
        ('analyst', 'Analyst'),
        ('admin', 'Admin')
    ])
    is_active = BooleanField('Active')
    submit = SubmitField('Update User')


class VerifyEmailForm(FlaskForm):
    """Email OTP verification form."""
    email = StringField('Email', validators=[
        DataRequired(message='Email is required.'),
        Regexp(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', message='Please enter a valid email address.')
    ])
    otp = StringField('Verification Code', validators=[
        DataRequired(message='Please enter the 6-digit verification code.'),
        Length(min=6, max=6, message='Verification code must be exactly 6 digits.'),
        Regexp(r'^\d{6}$', message='Verification code must contain only 6 numbers.')
    ])
    submit = SubmitField('Verify & Activate Account')

