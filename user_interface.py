import tkinter as tk
import ttkbootstrap as ttk
from predict_disease import load_models
from predict_disease import predict_disease

import sv_ttk

root = ttk.Window(themename="yeti")
root.title("Sintomask")
root.resizable(False, False)

# sv_ttk.use_light_theme()

# Global Options

DISEASES_AMOUNT = 5
global model_option, preprocessing_option 
global ner_model, classification_model
model_option = 'gru'
preprocessing_option = 1
ner_model, classification_model = load_models(model_option)

# Styles

#BG_COLOR = "#f5f5f5"
#INFO_COLOR = "#5bc0de"
# style = ttk.Style()
# style.configure('TLabelframe', background='#99d9d9')

def predict_using_input():
    symptoms, diseases = predict_disease(ner_model, classification_model, input_text.get(), preprocessing_option)

    #print(symptoms, diseases)

    # Get highest percentage disease
    sorted_disease_dict = {key: val for key, val in sorted(diseases.items(), key = lambda x: x[1], reverse = True)}

    disease_names = list(sorted_disease_dict)
    disease_probabilities = list(sorted_disease_dict.values())
    for i in range(len(disease_probabilities)):
        disease_probabilities[i] = round(disease_probabilities[i] * 100, 2)

    most_likely_disease = disease_names[0]
    most_likely_disease_percentage = disease_probabilities[0]

    # Change UI text
    all_symptoms = ""
    for i in range(len(symptoms)):
        if (i < len(symptoms) - 1):
            all_symptoms += symptoms[i] + ", "
        else:
            all_symptoms += symptoms[i]
    symptom_label.configure(text=all_symptoms)

    # Reset disease text first
    for i in range(len(disease_label_list)):
        if (i < DISEASES_AMOUNT):
            disease_label_list[i].configure(text="")

    for i in range(len(disease_names)):
        if (i < DISEASES_AMOUNT):
            disease_label_list[i].configure(text=(str(disease_names[i]).capitalize() + ": " + str(disease_probabilities[i]) + "%"))

    likely_disease_name.configure(text=most_likely_disease)
    likely_disease_meter.configure(amountused=most_likely_disease_percentage)
    disease_description_label.configure(text=get_disease_description(most_likely_disease))

def reload_model():
    global ner_model, classification_model
    ner_model, classification_model = load_models(model_option.get())

def get_disease_description(disease_name):
    import json

    with open("{}/{}.json".format('cfg', "disease_description"), encoding="utf8") as json_file:
        disease_description_list = json.load(json_file)
    
    if disease_name in disease_description_list:
        return disease_description_list[disease_name]
    else: 
        return "Disease description not found."

# Window Size
window_width = 900
window_height = 650

# get the screen dimension
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# find the center point
center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2)

# Root config
#root.configure(bg=BG_COLOR)
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

# Input
input_frame = ttk.Frame(root, bootstyle="primary")
input_frame.place(x=0, y=0, width=900, height=210)

subtitle = ttk.Label(root, text='Pakilagay ang iyong mga sintomas dito:',
                     bootstyle="inverse-primary", font=("Roboto", 18, "bold"))
subtitle.pack(side=tk.TOP, pady=50)

input_text = tk.StringVar()
inputText = ttk.Entry(root, textvariable=input_text, background="#aeeaea", foreground="#546464")
inputText.place(relx=0.45, rely=0.2, anchor=tk.CENTER, width=500, height=50)

button = ttk.Button(root, text="Itanong", bootstyle="info-outline", command=predict_using_input)
button.place(relx=0.81, rely=0.2, anchor=tk.CENTER, width=125, height=50)

# Most Likely Disease

likely_disease_label = ttk.Label(root, text="Ang iyong resulta ay:", bootstyle="primary", font=("Roboto", 24, "bold"))
likely_disease_label.place(x=70, y=285)

likely_disease_name = ttk.Label(root, wraplength=275, bootstyle="danger", font=("Roboto", 24,))
likely_disease_name.place(x=380, y=285)

likely_disease_meter = ttk.Meter(root,
                                bootstyle="danger", 
                                textright="%",
                                subtextstyle="danger",
                                subtext="posibilidad ng sakit",
                                textfont="-size 20 -weight bold",
                                subtextfont="-size 8",
                                metersize=150,
                                meterthickness=5,
                                stripethickness=10,
                                amountused=None
                                )
likely_disease_meter.place(relx=0.7, y=240)

# Symptoms Detected

symptom_frame = ttk.LabelFrame(root, text='Sintomas', labelanchor=tk.NW, bootstyle="info")
symptom_frame.place(relx=0.05, rely=0.925, anchor=tk.SW, width=175, height=200)

symptom_label = ttk.Label(symptom_frame, wraplength=125)
symptom_label.place(x=15, y=5)

# Disease List

disease_frame = ttk.LabelFrame(root, text='Mga Posibleng Sakit', labelanchor=tk.NW, bootstyle="info")
disease_frame.place(relx=0.28, rely=0.925, anchor=tk.SW, width=250, height=200)

disease_label_list = []
while (len(disease_label_list) < DISEASES_AMOUNT):
    disease_label = ttk.Label(disease_frame, wraplength=200)
    disease_label.place(x=15, y=5 + (len(disease_label_list) * 20))
    disease_label_list.append(disease_label)

# Disease Description

disease_description_frame = ttk.LabelFrame(root, text='Disease Description', labelanchor=tk.NW, bootstyle="info")
disease_description_frame.place(anchor=tk.SE, relx=0.925, rely=0.925, width=300, height=200)

disease_description_text = tk.StringVar()
disease_description_label = ttk.Label(disease_description_frame, wraplength=275)
disease_description_label.place(x=15, y=5)

root.mainloop()
