"""
Email Service Module (Brevo API)
================================
Handles sending transactional emails (Registration Confirmation, Alerts, Password Reset)
via Brevo API (v3) with fallback handling.
"""

import os
import json
import logging
import threading
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Try loading .env if environment variables are not already present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except Exception:
    pass

logger = logging.getLogger(__name__)

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"



def send_email_brevo(
    to_email: str,
    to_name: str,
    subject: str,
    html_content: str,
    text_content: Optional[str] = None,
    api_key: Optional[str] = None,
    sender_email: Optional[str] = None,
    sender_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send an email using Brevo REST API v3.
    
    Args:
        to_email: Recipient email address
        to_name: Recipient name / username
        subject: Email subject
        html_content: HTML body content
        text_content: Plain text body content fallback
        api_key: Brevo API key (defaults to env or config)
        sender_email: Sender email address
        sender_name: Sender name
        
    Returns:
        Dict with status and response info
    """
    key = api_key or os.environ.get('BREVO_API_KEY')
    from_email = sender_email or os.environ.get('BREVO_SENDER_EMAIL') or os.environ.get('GOOGLE_EMAIL', 'teamclickjack@gmail.com')
    from_name = sender_name or os.environ.get('BREVO_SENDER_NAME', 'AI-NIDS Security')
    
    if not key:
        logger.warning("Brevo API key not found. Email not sent.")
        return {'success': False, 'error': 'BREVO_API_KEY is not configured'}
        
    payload = {
        "sender": {
            "name": from_name,
            "email": from_email
        },
        "to": [
            {
                "email": to_email,
                "name": to_name
            }
        ],
        "subject": subject,
        "htmlContent": html_content
    }
    
    if text_content:
        payload["textContent"] = text_content
        
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        BREVO_API_URL,
        data=data,
        headers={
            "accept": "application/json",
            "api-key": key,
            "content-type": "application/json",
            "User-Agent": "AI-NIDS-Mailer/1.0"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            resp_data = response.read().decode('utf-8')
            resp_json = json.loads(resp_data) if resp_data else {}
            logger.info(f"Email sent successfully via Brevo to {to_email}. MessageId: {resp_json.get('messageId')}")
            return {'success': True, 'messageId': resp_json.get('messageId')}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8', errors='replace')
        logger.error(f"Brevo API error ({e.code}): {err_body}")
        return {'success': False, 'error': f"HTTP {e.code}: {err_body}"}
    except Exception as e:
        logger.error(f"Failed to send email via Brevo: {str(e)}")
        return {'success': False, 'error': str(e)}


def send_registration_confirmation_async(
    user_email: str,
    username: str,
    role: str = 'analyst'
) -> None:
    """
    Send registration confirmation / welcome email in a background thread
    so user registration remains instantaneous.
    """
    thread = threading.Thread(
        target=send_registration_confirmation,
        args=(user_email, username, role),
        daemon=True
    )
    thread.start()


def send_registration_confirmation(
    user_email: str,
    username: str,
    role: str = 'analyst'
) -> Dict[str, Any]:
    """
    Send a beautifully formatted registration confirmation email.
    """
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    subject = "🛡️ Welcome to AI-NIDS - Account Registration Confirmed"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0d1117;
                color: #c9d1d9;
                margin: 0;
                padding: 20px;
            }}
            .email-container {{
                max-width: 600px;
                margin: 0 auto;
                background: #161b22;
                border-radius: 12px;
                border: 1px solid #30363d;
                overflow: hidden;
                box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            }}
            .email-header {{
                background: linear-gradient(135deg, #1f6feb 0%, #1158c7 100%);
                padding: 30px;
                text-align: center;
                color: #ffffff;
            }}
            .email-header h1 {{
                margin: 0;
                font-size: 24px;
                letter-spacing: 0.5px;
            }}
            .email-header p {{
                margin: 8px 0 0 0;
                font-size: 14px;
                opacity: 0.9;
            }}
            .email-body {{
                padding: 30px;
                line-height: 1.6;
            }}
            .user-card {{
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }}
            .user-card table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .user-card td {{
                padding: 8px 0;
                border-bottom: 1px solid #21262d;
                font-size: 14px;
            }}
            .user-card td.label {{
                color: #8b949e;
                width: 40%;
                font-weight: 600;
            }}
            .user-card td.value {{
                color: #58a6ff;
                font-weight: bold;
            }}
            .btn-dashboard {{
                display: inline-block;
                background: #238636;
                color: #ffffff !important;
                text-decoration: none;
                padding: 12px 28px;
                border-radius: 6px;
                font-weight: 600;
                margin: 15px 0;
                text-align: center;
            }}
            .security-tips {{
                background: #1f242c;
                border-left: 4px solid #f0883e;
                padding: 15px;
                border-radius: 4px;
                margin-top: 20px;
                font-size: 13px;
            }}
            .email-footer {{
                background: #0d1117;
                padding: 20px;
                text-align: center;
                font-size: 12px;
                color: #8b949e;
                border-top: 1px solid #21262d;
            }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <div class="email-header">
                <h1>🛡️ AI-NIDS Security Platform</h1>
                <p>AI-Powered Network Intrusion Detection System</p>
            </div>
            <div class="email-body">
                <h2 style="color: #ffffff; margin-top: 0;">Welcome, {username}!</h2>
                <p>Your account on <strong>AI-NIDS Platform</strong> has been successfully created and registered.</p>
                
                <div class="user-card">
                    <table>
                        <tr>
                            <td class="label">Username:</td>
                            <td class="value">{username}</td>
                        </tr>
                        <tr>
                            <td class="label">Registered Email:</td>
                            <td class="value">{user_email}</td>
                        </tr>
                        <tr>
                            <td class="label">Assigned Role:</td>
                            <td class="value">{role.capitalize()}</td>
                        </tr>
                        <tr>
                            <td class="label">Registration Time:</td>
                            <td class="value" style="color: #c9d1d9;">{now_str}</td>
                        </tr>
                        <tr>
                            <td class="label">Status:</td>
                            <td class="value" style="color: #3fb950;">Active ✅</td>
                        </tr>
                    </table>
                </div>

                <div style="text-align: center; margin: 25px 0;">
                    <a href="http://127.0.0.1:5000/auth/login" class="btn-dashboard">Sign In to Dashboard →</a>
                </div>

                <div class="security-tips">
                    <strong>🔒 Security Reminder:</strong>
                    <ul style="margin: 5px 0 0 0; padding-left: 20px;">
                        <li>Never share your credentials with anyone.</li>
                        <li>Generate API Keys from your profile for programmatic API access.</li>
                    </ul>
                </div>
            </div>
            <div class="email-footer">
                <p>© {datetime.utcnow().year} AI-NIDS Platform. All rights reserved.</p>
                <p>Sent securely via Brevo Mailer</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_content = f"""
AI-NIDS Security Platform
=========================
Welcome, {username}!

Your account on AI-NIDS has been successfully created.

Account Details:
- Username: {username}
- Email: {user_email}
- Role: {role.capitalize()}
- Registration Date: {now_str}
- Status: Active

Sign In URL: http://127.0.0.1:5000/auth/login

Security Reminder:
Never share your password with anyone.

(c) {datetime.utcnow().year} AI-NIDS Platform.
"""
    
    return send_email_brevo(
        to_email=user_email,
        to_name=username,
        subject=subject,
        html_content=html_content,
        text_content=text_content
    )


def send_verification_otp_email_async(
    user_email: str,
    username: str,
    otp_code: str
) -> None:
    """
    Send OTP verification email asynchronously in a background thread.
    """
    thread = threading.Thread(
        target=send_verification_otp_email,
        args=(user_email, username, otp_code),
        daemon=True
    )
    thread.start()


def send_verification_otp_email(
    user_email: str,
    username: str,
    otp_code: str
) -> Dict[str, Any]:
    """
    Send a 6-digit OTP verification code for email confirmation upon registration.
    """
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    subject = f"🔑 {otp_code} is your AI-NIDS Verification Code"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0d1117;
                color: #c9d1d9;
                margin: 0;
                padding: 20px;
            }}
            .email-container {{
                max-width: 580px;
                margin: 0 auto;
                background: #161b22;
                border-radius: 12px;
                border: 1px solid #30363d;
                overflow: hidden;
                box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            }}
            .email-header {{
                background: linear-gradient(135deg, #1f6feb 0%, #0d419d 100%);
                padding: 28px;
                text-align: center;
                color: #ffffff;
            }}
            .email-header h1 {{
                margin: 0;
                font-size: 22px;
                font-weight: 700;
            }}
            .email-header p {{
                margin: 6px 0 0 0;
                font-size: 13px;
                opacity: 0.9;
            }}
            .email-body {{
                padding: 30px;
                line-height: 1.6;
            }}
            .otp-box {{
                background: #0d1117;
                border: 2px dashed #388bfd;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                margin: 25px 0;
            }}
            .otp-code {{
                font-size: 38px;
                font-weight: 800;
                letter-spacing: 10px;
                color: #58a6ff;
                font-family: 'Courier New', Courier, monospace;
            }}
            .otp-expiry {{
                font-size: 13px;
                color: #f0883e;
                margin-top: 8px;
                font-weight: 600;
            }}
            .email-footer {{
                background: #0d1117;
                padding: 20px;
                text-align: center;
                font-size: 12px;
                color: #8b949e;
                border-top: 1px solid #21262d;
            }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <div class="email-header">
                <h1>🛡️ Email Verification Required</h1>
                <p>AI-NIDS Platform Security Validation</p>
            </div>
            <div class="email-body">
                <p style="font-size: 16px; color: #ffffff; margin-top: 0;">Hello <strong>{username}</strong>,</p>
                <p>Thank you for registering on <strong>AI-NIDS Platform</strong>. Please enter the following 6-digit verification code to verify your email address and activate your account:</p>
                
                <div class="otp-box">
                    <div style="font-size: 12px; text-transform: uppercase; color: #8b949e; letter-spacing: 1.5px; margin-bottom: 6px;">Your One-Time Password (OTP)</div>
                    <div class="otp-code">{otp_code}</div>
                    <div class="otp-expiry">⏳ Code expires in 15 minutes</div>
                </div>

                <p style="font-size: 13px; color: #8b949e;">
                    Enter this code on the registration verification screen to complete your setup.
                </p>

                <div style="background: #1c2128; border-left: 4px solid #8b949e; padding: 12px; border-radius: 4px; margin-top: 20px; font-size: 12px; color: #8b949e;">
                    <strong>Note:</strong> If you did not create an account on AI-NIDS, please disregard this email. No action is required.
                </div>
            </div>
            <div class="email-footer">
                <p>© {datetime.utcnow().year} AI-NIDS Security System. All rights reserved.</p>
                <p>Sent securely via Brevo API</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_content = f"""
AI-NIDS Security Platform - Email Verification
==============================================
Hello {username},

Your 6-digit verification code is: {otp_code}

This code is valid for 15 minutes.
Enter this code on the verification screen to activate your account.

If you did not request this code, please ignore this email.

(c) {datetime.utcnow().year} AI-NIDS Security System.
"""
    
    return send_email_brevo(
        to_email=user_email,
        to_name=username,
        subject=subject,
        html_content=html_content,
        text_content=text_content
    )

