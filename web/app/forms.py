from django.contrib.auth import authenticate
from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

class SignupForm(forms.Form):  # Defining SignupForm as a subclass of django's forms.Form which allows this form class to inhertit the functionalities of django for handling forms
    # The parameters below defines the fields of this form
    username = forms.CharField(required=True, max_length=50, widget=forms.TextInput(attrs={'class':'form-input', 'id':'username'})) # Defining 'username' field which uses the 'TextInput' widget and specifying a class and id value to it using the HTML attributes dictionary for styling it
    first_name = forms.CharField(required=True, max_length=50, widget=forms.TextInput(attrs={'class':'form-input', 'id':'first_name'})) # Defining 'first_name' field which uses the 'TextInput' widget and specifying a class and id value to it using the HTML attributes dictionary for styling it
    last_name = forms.CharField(required=True, max_length=50, widget=forms.TextInput(attrs={'class':'form-input', 'id':'last_name'})) # Defining 'last_name' field which uses the 'TextInput' widget and specifying a class and id value to it using the HTML attributes dictionary for styling it
    email = forms.EmailField(required=True, max_length=50, widget=forms.EmailInput(attrs={'class':'form-input', 'id':'email'})) # Defining 'email' field which uses the 'EmailInput' widget to add email field validation and specifying a class and id value to it using the HTML attributes dictionary for styling it
    password = forms.CharField(required=True, max_length=50, widget=forms.PasswordInput(attrs={'class':'form-input', 'id':'password'})) # Defining 'password' field which uses the 'PasswordInput' widget to codify the password and specifying a class and id value to it using the HTML attributes dictionary for styling it
    confirm_password = forms.CharField(required=True, max_length=50, widget=forms.PasswordInput(attrs={'class':'form-input', 'id':'password_change'})) # Defining 'confirm_password' field which uses the 'PasswordInput' widget to codify the password and specifying a class and id value to it using the HTML attributes dictionary for styling it

    def clean(self): # This function will run when we will call form.is_valid() in the view function of this form. 
        # Specifing the non-field validation criteria of this form:
        if self.cleaned_data['password'] != self.cleaned_data['confirm_password']: # If the cleaned confirmation password does not match, then a validation error is raised
            raise ValidationError('Both passwords are not matching!')
        if User.objects.filter(username=self.cleaned_data['username']).exists(): # If the entered username already exists in the database table that the 'User' model represents (using ORM), then a validation error is raised
            raise ValidationError('The username is already used! Please try another one')
        
    def save(self): #This function will run when we will call form.save() in the view function of this form.
        # The data stored in the fields of the form are inserted into the record created for the new user in the database table that the 'User' model represents (using ORM)
        new_record = User.objects.create_user(username=self.cleaned_data['username'], email=self.cleaned_data['email'], password=self.cleaned_data['password'])
        new_record.first_name = self.cleaned_data['first_name']
        new_record.last_name = self.cleaned_data['last_name']
        new_record.save() # The new record is saved to the database table that the 'User' model represents
        return new_record # The new record is returned

class LoginForm(forms.Form):  # Defining LoginForm as a subclass of django's forms.Form which allows this form class to inhertit the functionalities of django for handling forms
    # The parameters below defines the fields of this form
    username = forms.CharField(required=True, max_length=50, widget=forms.TextInput(attrs={'class':'form-input', 'id':'username'}))  # Defining 'username' field which uses the 'TextInput' widget and specifying a class and id value to it using the HTML attributes dictionary for styling it
    password = forms.CharField(required=True, max_length=50, widget=forms.PasswordInput(attrs={'class':'form-input', 'id':'password'})) # Defining 'password' field which uses the 'PasswordInput' widget to codify the password and specifying a class and id value to it using the HTML attributes dictionary for styling it

    def clean(self): # This function will run when we will call form.is_valid() in the view function of this form. 
        # Specifing the non-field validation criteria of this form:
        cleaned_data = self.cleaned_data # Storing the cleaned form of the form's inputs
        username = cleaned_data['username'] # Obtaining the cleaned form of the username entered by the user
        password = cleaned_data['password'] # Obtaining the cleaned form of the password entered by the user
        user = authenticate(username=username, password=password) # Verifying the user's username and pasword with the ones stored Django authentication system
        if not user: # If the user is not fails authentication, then a validation error is raised
            raise ValidationError('Username or password is incorrect! Please Try again')
        return cleaned_data # If the validation is successful, the cleaned data is retured

class EditProfileForm(forms.ModelForm):  # Defining EditProfileForm as a subclass of django's forms.ModelForm which allows this form class to directly interact with the database models
    class Meta:
        model = User # The fields and behaviours of the this form will be based on the ones defined in the 'User' model
        fields = ['username', 'first_name', 'last_name', 'email'] # Specifying the fields of the 'User' model that will be used in this form
        widgets = {
            'username': forms.TextInput(attrs={'class':'form-input', 'id':'username'}), # Defining that the 'username' field uses the 'TextInput' widget and specifying a class and id value to it using the HTML attributes dictionary for styling it
            'first_name': forms.TextInput(attrs={'class':'form-input', 'id':'first_name'}), # Defining that the 'first_name' field uses the 'TextInput' widget and specifying a class and id value to it using the HTML attributes dictionary for styling it
            'last_name': forms.TextInput(attrs={'class':'form-input', 'id':'last_name'}), # Defining that the 'last_name' field uses the 'TextInput' widget and specifying a class and id value to it using the HTML attributes dictionary for styling it
            'email': forms.EmailInput(attrs={'class':'form-input', 'id':'email'}), # Defining that the 'email' field uses the 'EmailInput' widget to add email field validation and specifying a class and id value to it using the HTML attributes dictionary for styling it
        }

    def clean(self): # This function will run when we will call form.is_valid() in the view function of this form. 
        # Specifing the non-field validation criteria of this form:
        if User.objects.filter(username=self.cleaned_data['username']).exists(): # If the entered username already exists in the database table that the 'User' model represents (using ORM), then a validation error is raised
            raise ValidationError('The username is already used! Please try another one') 
    
    def save(self): #This function will run when we will call form.save() in the view function of this form.
        user_record = self.instance # An instance of the values entered in this form will be created and stored, which represents a record of the user in the database table of the 'User' model
        user_record.save() # The edited record is saved to the database table that the 'User' model represents
        return user_record # The edited record is returned
    

class PasswordChange(forms.Form): # Defining PasswordChange form as a subclass of django's forms.Form which allows this form class to inhertit the functionalities of django for handling forms
    old_password = forms.CharField(required=True, max_length=50, widget=forms.PasswordInput(attrs={'class':'form-input', 'id':'old_password'})) # Defining 'old_password' field which uses the 'PasswordInput' widget to codify the password and specifying a class and id value to it using the HTML attributes dictionary for styling it
    new_password = forms.CharField(required=True, max_length=50, widget=forms.PasswordInput(attrs={'class':'form-input', 'id':'new_password'})) # Defining 'new_password' field which uses the 'PasswordInput' widget to codify the password and specifying a class and id value to it using the HTML attributes dictionary for styling it
    confirm_new_password = forms.CharField(required=True, max_length=50, widget=forms.PasswordInput(attrs={'class':'form-input', 'id':'confirm_new_password'})) # Defining 'confirm_new_password' field which uses the 'PasswordInput' widget to codify the password and specifying a class and id value to it using the HTML attributes dictionary for styling it

    def __init__(self, user, *args, **kwargs): # This function runs when the form is being initialized
        self.user = user
        super(PasswordChange, self).__init__(*args, **kwargs) # This ensures that any arguments passed to the PasswordChange form's constructor are correctly forwarded to the parent class

    def clean(self): # This function will run when we will call form.is_valid() in the view function of this form. 
        # Specifing the validation criteria of this form:
        old_password = self.cleaned_data['old_password']
        if not self.user.check_password(old_password): # If the entered old password is incorrect, then a validation error is raised
            raise ValidationError('The old password is incorrect! Please try again')
        if self.cleaned_data['new_password'] != self.cleaned_data['confirm_new_password']: # If the confirmation password does not match, then a validation error is raised
            raise ValidationError('The new passwords are not matching! Please try again')
    
    def save(self): #This function will run when we will call form.save() in the view function of this form.
        self.user.set_password(self.cleaned_data['new_password']) # This updates user's old password to the new one
        self.user.save() # This saves the updated password 
        return self.user # The current user object is returned

class AiModelForm(forms.Form):  # Defining AiModelForm as a subclass of django's forms.Form which allows this form class to inhertit the functionalities of django for handling forms
    crypto_options = [
        ('BTC-USD','Bitcoin'),
        ('ETH-USD','Ethereum'),
        ('DOGE-USD','Doge coin'),
    ] # The first element represents the actual cryptocurrency's code name that will be stored in this form, and the second element represents the human-readable name of the cryptocurrency that will be displayed in the form for clarity
    days_options = [
        ('5','5 Days'),
        ('10','10 Days'),
        ('15','15 Days'),
    ] # The first element represents the actual number of days that will be stored in this form, and the second element represents the human-readable number of days that will be displayed in the form for clarity
    cryptocurrency = forms.ChoiceField(required=True, choices=crypto_options , widget=forms.Select(attrs={'class':'form-input', 'id':'cryptocurrency'})) # Defining 'cryptocurrency' field which is a 'ChoiceField' that adds its validation criteria to this field and uses the 'Select' widget for creating a dropdown list of the choices and specifying a class and id value to it using the HTML attributes dictionary for styling it
    n_ahead = forms.ChoiceField(required=True, choices=days_options , widget=forms.Select(attrs={'class':'form-input', 'id':'n_ahead'})) # Defining 'n_ahead' field which is a 'ChoiceField' that adds its validation criteria to this field and uses the 'Select' widget for creating a dropdown list of the choices and specifying a class and id value to it using the HTML attributes dictionary for styling it
    def clean(self): # This function will run when we will call form.is_valid() in the view function of this form. 
        cleaned_data = self.cleaned_data 
        return cleaned_data # cleaned data is returned when the form.is_valid() is called on this form in the views.py
    