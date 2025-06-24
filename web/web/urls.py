from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('', views.home_view, name="home"),# When user tries to access the base url, the home_view function is called. This url pattern is named "docs"
    path('docs/', views.docs_view, name="docs"), # When user tries to access the 'docs/' url, the docs_view function is called. This url pattern is named "docs"
    path('model/', views.ai_model_view, name="model"), # When user tries to access the 'model/' url, the ai_model_view function is called. This url pattern is named "model"
    path('signup/', views.signup_view, name='signup'), # When user tries to access the 'signup/' url, the signup_view function is called. This url pattern is named "signup"
    path('login/', views.login_view, name='login'), # When user tries to access the 'login/' url, the login_view function is called. This url pattern is named "login"
    path('logout/', views.logout_view, name='logout'), # When user tries to access the 'logout/' url, the logout_view function is called. This url pattern is named "logout"
    path('password_change/', views.password_change_view, name='password_change'), # When user tries to access the 'password_change/' url, the password_change_view function is called. This url pattern is named "password_change"
    path('profile/', views.profile_view, name="profile"), # When user tries to access the 'profile/' url, the profile_view function is called. This url pattern is named "profile_view"
    path('edit_profile/', views.edit_profile_view, name='edit_profile'), # When user tries to access the 'edit_profile/' url, the edit_profile_view function is called. This url pattern is named "edit_profile"
    path('admin/', admin.site.urls), # The default AdminSite instance of the webapp is registered at 'admin/' url
]
