from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

diseases=['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']

symptoms=[]

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        itching=int(request.form['itching'])
        skin_rash=int(request.form['skin_rash'])
        continuous_sneezing=int(request.form['continuous_sneezing'])
        chills=int(request.form['chills'])
        joint_pain=int(request.form['joint_pain'])
        vomiting=int(request.form['vomiting'])
        fatigue=int(request.form['fatigue'])
        weight_loss=int(request.form['weight_loss'])
        cough=int(request.form['cough'])
        high_fever=int(request.form['high_fever'])
        headache=int(request.form['headache'])
        yellowish_skin=int(request.form['yellowish_skin'])
        signs=[itching,skin_rash,continuous_sneezing,chills,joint_pain,vomiting,fatigue,weight_loss,cough,high_fever,headache,yellowish_skin]
        features=np.array([signs])
        pred=model.predict(features)
        disease=diseases[pred[0]-1]
        return render_template('index.html',disease=disease)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
    