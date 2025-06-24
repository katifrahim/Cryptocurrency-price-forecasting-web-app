from django.db import models
from django.contrib.auth.models import User

class User_Logs(models.Model): # Defining User_Logs as a subclass of django's forms.Model which allows this class to inhertit the functionalities of django models. This model represents a database table
    user_id = models.ForeignKey(User, on_delete=models.CASCADE) # Defining user_id field as a ForeignKey field for this model. If user id of a specific user is deleted from the User model then all the records containing the user id as foreign key in this model will also be deleted.
    input_coin = models.CharField(max_length=50) # Defining input_coin field as a CharField containing values which should have a max length of 50 characters
    input_days = models.CharField(max_length=50) # Defining input_days field as a CharField containing values which should have a max length of 50 characters
    output_trend = models.CharField(max_length=50) # Defining output_trend field as a CharField containing values which should have a max length of 50 characters
    date = models.DateField() # Defining date field as a DateField containing values with datetime data type
    class Meta:
        db_table = 'user_logs' # Defining the name of the database table that this model represents


