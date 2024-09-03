from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.validators import FileExtensionValidator
# Create your models here.

class Topic(models.Model):
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name
    


class Doctor(models.Model):
    name = models.CharField(max_length=100)
    designation = models.CharField(max_length=100)
    licence = models.ImageField(upload_to='ehr/static', validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png'])], null=True)
    phonenumber = models.CharField(max_length=100)
    address = models.TextField(null=True, blank=True)
    emailid = models.EmailField()

    def __str__(self):
        return self.name

class Individual(models.Model):
    name = models.CharField(max_length=20)
    gender = models.CharField(max_length=10, null=True)
    age = models.IntegerField(default=0)
    DOB = models.DateField(default=timezone.now)
    address = models.TextField(null=True, blank=True)
    phonenumber = models.CharField(max_length=10)

# ehrapp/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    face_encoding = models.BinaryField(null=True, blank=True)

# If you are using a custom user model, don't forget to update the AUTH_USER_MODEL in settings.py
# AUTH_USER_MODEL = 'ehrapp.CustomUser'

# ehrapp/models.py

from django.db import models
from torch import nn

class CNNModel(nn.Module):
    # Your CNNModel implementation here


    def __str__(self):
        return self.name
    


