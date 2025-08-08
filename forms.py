from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, SubmitField
from wtforms.validators import InputRequired, Length, EqualTo, ValidationError

class CaseInsensitiveStringField(StringField):
    def process_formdata(self, valuelist):
        super().process_formdata(valuelist)
        if self.data:
            self.data = self.data.lower()

class LoginForm(FlaskForm):
    username = CaseInsensitiveStringField('Username', validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8)])

class ChangePasswordForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[InputRequired()])
    new_password = PasswordField('New Password', validators=[InputRequired(), Length(min=8)])

class RegisterForm(FlaskForm):
    """
    Enhanced registration form with clearance level selection and admin code verification.
    """
    username = CaseInsensitiveStringField('Username', validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[
        InputRequired(), EqualTo('password', message='Passwords must match')
    ])
    
    # NEW: Clearance level selection
    clearance_level = SelectField(
        'Clearance Level',
        choices=[('guest', 'User'), ('admin', 'Admin')],
        default='guest',
        validators=[InputRequired()]
    )
    
    # NEW: Admin verification code field
    admin_code = StringField('Admin Verification Code')
    
    submit = SubmitField('Register')

    def validate_admin_code(self, field):
        """
        Custom validator for admin code verification.
        Only validates when user selects Admin clearance level.
        Ensures the admin code is exactly '12345'.
        """
        if self.clearance_level.data == 'admin':
            if not field.data:
                raise ValidationError('Admin code is required for Admin access.')
            if field.data.strip() != '12345':
                raise ValidationError('Invalid admin code.')