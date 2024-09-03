from django.forms import ModelForm
from .models import Doctor

class DoctorForm(ModelForm):
    class Meta:
        model = Doctor
        fields = '__all__'
        exclude = ['host','participants']
# ehrapp/forms.py
from django import forms

class UserForm(forms.Form):
    username = forms.CharField(max_length=100)
