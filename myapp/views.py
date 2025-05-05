from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import seaborn as sn
#BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load('model.h5')

def index(request):

    if request.method == "POST":
        loginusername = request.POST.get("loginusername")
        loginpassword = request.POST.get("loginpassword")

        if loginusername == "admin" and loginpassword == "admin":
            return redirect('dashboard')
        else:
            return render(request, "index.html",context = {"mymessage":"Please check your password.","Flag":"True", "Navigate":"False"}) 

    return render(request, "index.html")

def home(request):
    if request.method == "POST":

        duration = request.POST.get('duration')
        protocol_type  = request.POST.get('protocol_type')
        flag  = request.POST.get('flag')
        src_bytes  = request.POST.get('src_bytes')
        dst_bytes  = request.POST.get('dst_bytes')
        land  = request.POST.get('land')
        wrong_fragment  = request.POST.get('wrong_fragment')
        urgent  = request.POST.get('urgent')
        hot  = request.POST.get('hot')
        num_failed_logins  = request.POST.get('num_failed_logins')
        logged_in  = request.POST.get('logged_in')
        num_compromised  = request.POST.get('num_compromised')

        testing_data = []

        testing_data.append(duration)
        testing_data.append(protocol_type)
        testing_data.append(flag)
        testing_data.append(src_bytes)
        testing_data.append(dst_bytes)
        testing_data.append(land)

        testing_data.append(wrong_fragment)
        testing_data.append(urgent)
        testing_data.append(hot)
        testing_data.append(num_failed_logins)
        testing_data.append(logged_in)
        testing_data.append(num_compromised)

        print(testing_data)

        request.session['testing_data'] = testing_data

        return redirect("home_2")

    return render(request, "home.html")

def home_2(request):

    if request.method == "POST":
        root_shell = request.POST.get('root_shell')
        su_attempted  = request.POST.get('su_attempted')
        num_file_creations  = request.POST.get('num_file_creations')
        num_shells  = request.POST.get('num_shells')
        num_access_files  = request.POST.get('num_access_files')
        is_guest_login  = request.POST.get('is_guest_login')
        count  = request.POST.get('count')
        srv_count  = request.POST.get('srv_count')
        serror_rate  = request.POST.get('serror_rate')
        rerror_rate  = request.POST.get('rerror_rate')
        same_srv_rate  = request.POST.get('same_srv_rate')
        diff_srv_rate  = request.POST.get('diff_srv_rate')

        testing_data = request.session['testing_data']
        print(testing_data)

        testing_data.append(root_shell)
        testing_data.append(su_attempted)
        testing_data.append(num_file_creations)
        testing_data.append(num_shells)
        testing_data.append(num_access_files)
        testing_data.append(is_guest_login)

        testing_data.append(count)
        testing_data.append(srv_count)
        testing_data.append(serror_rate)
        testing_data.append(rerror_rate)
        testing_data.append(same_srv_rate)
        testing_data.append(diff_srv_rate)

        print(testing_data)
        request.session['testing_data'] = testing_data

        return redirect("home_3")

        

    return render(request, "home_2.html")

def home_3(request):

 
    if request.method == "POST":

        srv_diff_host_rate = request.POST.get('srv_diff_host_rate')
        dst_host_count  = request.POST.get('dst_host_count')
        dst_host_srv_count  = request.POST.get('dst_host_srv_count')
        dst_host_diff_srv_rate  = request.POST.get('dst_host_diff_srv_rate')
        dst_host_same_src_port_rate  = request.POST.get('dst_host_same_src_port_rate')
        dst_host_srv_diff_host_rate  = request.POST.get('dst_host_srv_diff_host_rate')

        testing_data = request.session['testing_data']
        print(testing_data)

        testing_data.append(srv_diff_host_rate)
        testing_data.append(dst_host_count)
        testing_data.append(dst_host_srv_count)
        testing_data.append(dst_host_diff_srv_rate)
        testing_data.append(dst_host_same_src_port_rate)
        testing_data.append(dst_host_srv_diff_host_rate)

        request.session['testing_data'] = testing_data
        print(testing_data)

        
        arr = np.array(testing_data)
        pred = model.predict(arr.reshape(1,-1))

        return render(request, "home_3.html",context = {"mymessage":"Detected Attack : " + pred[0],"Flag":"True", "Navigate":"False"}) 

    return render(request, "home_3.html")

def dashboard(request):

    if request.method == "POST":
        return redirect('home')

 
    path = "dataset.csv"
    df = pd.read_csv(path)

    dos = 0
    normal = 0
    probe = 0
    r2l = 0
    u2r = 0

    for i in df['Attack Type']:
      if i == "dos":
        dos+=1
      if i == "normal":
        normal+=1
      if i == "probe":
        probe+=1
      if i == "r2l":
        r2l+=1
      if i == "u2r":
        u2r+=1


    context = {'dos': dos,
               'normal': normal,
               'probe': probe,
               'r2l': r2l,
               'u2r': u2r}

    return render(request=request,
                      template_name='dashboard.html',
                      context=context)