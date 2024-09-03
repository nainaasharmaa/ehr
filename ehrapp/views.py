from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from .models import Topic, Doctor, Individual
from .forms import DoctorForm
from django.db.models import Q
import torch.nn as nn
import os
import cv2
import numpy as np
from .models import CustomUser
from .forms import UserForm
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
from django.conf import settings
from django.contrib.auth import login as auth_login

class CNNModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model_path = os.path.join(settings.BASE_DIR, 'face_recognition_model.pth')
num_classes = 2
model = CNNModel(num_classes)
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    print(f"Model file not found at {model_path}")

def home(request):
    return render(request, 'home.html', {})

@login_required(login_url='login')
def DoctorProfile(request, pk):
     doctor = Doctor.objects.get(id=pk)
     context = {'doctor': doctor}
     return render(request, 'ehrapp/doctor.html', context)

@login_required(login_url='login')
def IndividualProfile(request,name):
    individual = Individual.objects.get(name=name)
    context = {'individual':individual}
    return render(request,'ehrapp/individual.html',context)

def notexist(request):
    return render(request, 'ehrapp/notexist.html', {})

def search(request):
    query = request.GET.get('Individual')
    if query:
        allPosts = Individual.objects.filter(title__icontains=query)
    else:
        allPosts = []
    context = {'allPosts': allPosts}
    return render(request, 'ehrapp/search.html', context)

def register(request):
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            video_capture = cv2.VideoCapture(0)
            ret, frame = video_capture.read()
            video_capture.release()
            cv2.destroyAllWindows()
            if not ret:
                return render(request, 'register.html', {'form': form, 'error': 'Failed to capture image'})

            photo_path = os.path.join("C:\\Users\\Naina Sharma\\Desktop\\Electronic-Health-Record-System\\faces\\1.jpg", f'{username}.jpg')

            cv2.imwrite(photo_path, frame)

            pil_image = Image.fromarray(frame)
            transformed_image = transform(pil_image)
            face_encoding = transformed_image.numpy().tobytes()

            user = CustomUser(username=username, face_encoding=face_encoding)
            user.set_password(form.cleaned_data['password'])
            user.save()
            return redirect('login')
    else:
        form = UserForm()
    return render(request, 'register.html', {'form': form})

def face_login(request):
    if request.method == 'POST':
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        video_capture.release()
        cv2.destroyAllWindows()
        if not ret:
            return render(request, 'login.html', {'error': 'Failed to capture image'})

        pil_image = Image.fromarray(frame)
        transformed_image = transform(pil_image)
        face_encoding = transformed_image.numpy().tobytes()

        users = CustomUser.objects.all()
        for user in users:
            if user.face_encoding:
                stored_face_encoding = np.frombuffer(user.face_encoding, dtype=np.float32).reshape(3, 224, 224)
                stored_face_encoding_tensor = torch.tensor(stored_face_encoding, dtype=torch.float32).unsqueeze(0)
                face_encoding_tensor = torch.tensor(transformed_image.numpy(), dtype=torch.float32).unsqueeze(0)

                output = model(stored_face_encoding_tensor)
                _, predicted = torch.max(output.data, 1)

                output_login = model(face_encoding_tensor)
                _, predicted_login = torch.max(output_login.data, 1)

                if predicted.item() == predicted_login.item():
                    auth_login(request, user)
                    return redirect('dashboard')
        return render(request, 'login.html', {'error': 'User not recognized'})
    return render(request, 'login.html')

@login_required
def dashboard(request):
    return render(request, 'dashboard.html')
