from django.contrib import admin
from .models import User_Logs
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User, Group

# Registering models here:
class User_Logs_Admin(admin.ModelAdmin): # Defining the User_Logs_Admin class which is a subclass of django's admin.ModelAdmin
    list_display=('id', 'user_id', 'input_coin', 'input_days', 'output_trend', 'date') # Listing the fields of the User Logs model that should be displayed in the admin site

class User_Admin(UserAdmin): # Adding the phone number and active field to the UserAdmin model
    list_display=('username', 'first_name', 'last_name', 'email', 'is_staff', 'is_active', 'is_superuser', 'last_login', 'date_joined') # Listing the fields of the User model that should be displayed in the admin site

# Registering and unregistering models to the admin site
admin.site.register(User_Logs, User_Logs_Admin) # Registering the User_Logs model with customized User_Logs_Admin class to the admin site to be displayed
admin.site.unregister(User) # Unregistering the by-default User model from the admin site
admin.site.unregister(Group) # Unregistering the by-default Group model from the admin site
admin.site.register(User, User_Admin) # Registering the User model with customized User_Admin class to the admin site to be displayed
