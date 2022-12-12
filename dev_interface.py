import tkinter as tk
from tkinter import ttk
from predict_disease import load_models
from predict_disease import predict_disease

import sv_ttk

root = tk.Tk()
root.title("Sintomask")
root.resizable(False, False)

sv_ttk.use_light_theme()

DISEASES_AMOUNT = 5
global ner_model, classification_model
ner_model, classification_model = load_models('lstm')

def predict_using_input():
    symptoms, diseases = predict_disease(ner_model, classification_model, input_text.get(), preprocessing_option.get())

    #print(symptoms, diseases)

    symptom_text = ""
    disease_text = []
    if (len(symptoms) > 0):
        # Get all symptoms
        for i in range(len(symptoms)):
            if (i < len(symptoms) - 1):
                symptom_text += symptoms[i] + ", "
            else:
                symptom_text += symptoms[i]

        # Get highest percentage disease
        sorted_disease_dict = {key: val for key, val in sorted(diseases.items(), key = lambda x: x[1], reverse = True)}

        disease_names = list(sorted_disease_dict)
        disease_list_len = len(disease_names)
        disease_probabilities = list(sorted_disease_dict.values())
        for i in range(disease_list_len):
            disease_probabilities[i] = round(disease_probabilities[i] * 100, 2)
            disease_text.append(str(disease_names[i]).capitalize() + ": " + str(disease_probabilities[i]) + "%")

        most_likely_disease = disease_names[0]
        most_likely_disease_percentage = disease_probabilities[0]
    else:
        disease_list_len = 1
        disease_text.append("No diseases predicted!")
        most_likely_disease = "None"

    # Reset text
    for i in range(len(disease_label_list)):
        if (i < DISEASES_AMOUNT):
            disease_label_list[i].configure(text="")

    # Change UI text
    symptom_label.configure(text=symptom_text)

    for i in range(disease_list_len):
        if (i < DISEASES_AMOUNT):
            disease_label_list[i].configure(text=(disease_text[i]))

    likely_disease_name.configure(text=most_likely_disease)
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
window_height = 600

# get the screen dimension
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# find the center point
center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2)

# set the position of the window to the center of the screen
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

# Input
subtitle = ttk.Label(root, text='Pakilagay ang iyong mga sintomas dito:')
subtitle.pack(side=tk.TOP, pady=100)

input_text = tk.StringVar()
inputText = ttk.Entry(root, textvariable=input_text)
inputText.place(relx=0.45, rely=0.3, anchor=tk.CENTER, width=500, height=50)

button = ttk.Button(root, text="Itanong", command=predict_using_input)
button.place(relx=0.81, rely=0.3, anchor=tk.CENTER, width=125, height=50)

# Options

label_model_option = ttk.Label(root, text='Model:')
label_model_option.place(x=550, y=230)

label_preprocessing_option = ttk.Label(root, text='Preprocessing:')
label_preprocessing_option.place(x=505, y=270)

model_option = tk.StringVar()
model_option.set('lstm')
rb_model_lstm = ttk.Radiobutton(root, text='LSTM-based', value='lstm', variable=model_option, command=reload_model)
rb_model_lstm.place(x=600, y=225)

rb_model_gru = ttk.Radiobutton(root, text='GRU-based', value='gru', variable=model_option, command=reload_model)
rb_model_gru.place(x=720, y=225)

preprocessing_option = tk.IntVar()
preprocessing_option.set(1)
rb_preprocess = ttk.Radiobutton(root, text='Enable', variable=preprocessing_option, value=1)
rb_preprocess.place(x=600, y=265)

rb_preprocess_disable = ttk.Radiobutton(root, text='Disable', variable=preprocessing_option, value=0)
rb_preprocess_disable.place(x=720, y=265)

# Most Likely Disease

likely_disease_label = ttk.Label(root, text="Most Likely Disease: ")
likely_disease_label.place(x=100, y=265)

likely_disease_name = ttk.Label(root, wraplength=275)
likely_disease_name.place(x=250, y=265)

# Symptoms Detected

symptom_frame = ttk.LabelFrame(root, text='Sintomas', labelanchor=tk.NW)
symptom_frame.place(relx=0.05, rely=0.925, anchor=tk.SW, width=175, height=200)

symptom_label = ttk.Label(symptom_frame, wraplength=125)
symptom_label.place(x=15, y=5)

# Disease List

disease_frame = ttk.LabelFrame(root, text='Mga Posibleng Sakit', labelanchor=tk.NW)
disease_frame.place(relx=0.28, rely=0.925, anchor=tk.SW, width=250, height=200)

disease_label_list = []
while (len(disease_label_list) < DISEASES_AMOUNT):
    disease_label = ttk.Label(disease_frame, wraplength=200)
    disease_label.place(x=15, y=5 + (len(disease_label_list) * 20))
    disease_label_list.append(disease_label)

# Disease Description

disease_description_frame = ttk.LabelFrame(root, text='Disease Description', labelanchor=tk.NW)
disease_description_frame.place(anchor=tk.SE, relx=0.925, rely=0.925, width=300, height=200)

disease_description_text = tk.StringVar()
disease_description_label = ttk.Label(disease_description_frame, wraplength=275)
disease_description_label.place(x=15, y=5)

root.mainloop()
