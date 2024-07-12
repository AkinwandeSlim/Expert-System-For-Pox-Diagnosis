import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Updated Dataset
dataset = [
    (['Fever','Headache','Fatigue','Rash','Loss of Appetite','Sore Throat','Fluid-Filled Blisters'], 'Chickenpox'),
    (['High Fever','Cough', 'Rash', 'Red Watery Eyes','Running Noise','Sore Throat','Fatigue','Loss of Appetite'], 'Measles'),
    (['Fever','Headache','Bodyache','Swollen Lymph Nodes','Chills','Fatigue','Rash'], 'Monkepox'),
    (['High Fever','Fatigue','Severe Headache','Bodyache','Vomiting','Rash','Fluid-Filled Blisters'], 'Smallpox'),
    
]
# Separate symptoms and labels
symptoms = [data[0] for data in dataset]
labels = [data[1] for data in dataset]

# Create symptom vocabulary
symptom_vocabulary = list(set(symptom for symptoms in symptoms for symptom in symptoms))

# Convert symptoms into numerical features using binary indicators
def convert_to_features(data):
    features = []
    for symptoms in data:
        symptom_counts = {symptom: 1 if symptom in symptoms else 0 for symptom in symptom_vocabulary}
        symptom_vector = list(symptom_counts.values())
        features.append(symptom_vector)
    return np.array(features)



# Convert symptoms into numerical features using frequency indicators
def convert_to_features_freq(data):
    features = []
    for symptoms in data:
        symptom_counts = {symptom: symptoms.count(symptom) for symptom in symptom_vocabulary}
        symptom_vector = list(symptom_counts.values())
        features.append(symptom_vector)
    return np.array(features)






# Convert training and testing data into numerical features
features = convert_to_features_freq(symptoms)

# Initialize and train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(features, labels)

# Function to predict disease and visualize results
def predict_disease(symptoms, classifier, labels):
    test_features = convert_to_features_freq([symptoms])
    predicted_label_index = np.argmax(classifier.predict_proba(test_features))
    predicted_label = labels[predicted_label_index]
    predicted_probabilities = classifier.predict_proba(test_features)[0]

    # Sort the probabilities and labels in descending order
    sorted_indices = np.argsort(predicted_probabilities)[::-1]
    sorted_labels = np.array(labels)[sorted_indices]
    sorted_probabilities = predicted_probabilities[sorted_indices]

    # Plot the results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar Chart
    axes[0].barh(sorted_labels, sorted_probabilities)
    axes[0].set_xlabel('Probability')
    axes[0].set_ylabel('Disease')
    axes[0].set_title('Bar Chart: Predicted Disease Probabilities')

    # Pie Chart
    axes[1].pie(sorted_probabilities, labels=sorted_labels, autopct='%1.1f%%')
    axes[1].set_title('Pie Chart: Predicted Disease Probabilities')

    plt.tight_layout()

    # Display the plots in Streamlit
    st.pyplot(fig)

  # Print the predicted disease
    st.header(f"Predicted Disease: {predicted_label}")




# # Page 1: Home
# # def home():
#     st.title("Diagnostic Expert System for Pox-related Diseases")
#     st.write("Enter the symptoms below:")
#     # User input for symptoms
#     user_symptoms = st.multiselect("Select Symptoms", symptom_vocabulary)

# Page 1: Home
def home():
    st.title("Diagnostic System for Pox-related Diseases")
    st.write("Enter the symptoms below:")
    # User input for symptoms
    user_symptoms = st.multiselect("Select Symptoms", symptom_vocabulary)

    if st.button("Diagnose"):
        if len(user_symptoms) > 0:
            predict_disease(user_symptoms, classifier, labels)
        else:
            st.warning("Please select at least one symptom.")
            
# Page 2: Data and Mapping
def data_mapping():
    st.title("Data and Mapping")
    st.write("Here is the list of data and how they map to diseases:")

    # Create a DataFrame for data and mapping
    df = pd.DataFrame(dataset, columns=["Symptoms", "Disease"])
    df.index += 1  # Start index from 1

    # Display the DataFrame
    st.dataframe(df)






# Page 3: Add Symptoms and Diseases
def add_symptoms_and_diseases():
    st.title("Add Symptoms and Diseases")
    st.write("Enter the symptoms and corresponding disease below:")

    # User input for symptoms and disease
    user_symptoms = st.multiselect("Enter Symptoms", symptom_vocabulary)
    user_disease = st.text_input("Enter Disease")

    if st.button("Add"):
        if len(user_symptoms) > 0 and user_disease != "":
            # Add the new symptoms and disease to the dataset
            dataset.append((user_symptoms, user_disease))
            st.success("Symptoms and Disease added successfully!")
        else:
            st.warning("Please enter at least one symptom and disease.")






# Page 4: Add Symptoms
def Diseases_info():
    st.title("Info about Diseases")
    # st.write("select diseases from list")

    # User input for symptoms
    predicted_label = st.selectbox('Select Diseases',labels)
  
    if predicted_label == 'Chickenpox':
    	st.image('chicken-pox.png')
    	st.subheader('Chickenpox (Varicella):')
    	st.write("Chickenpox is a highly contagious viral illness caused by the varicella-zoster virus. It is characterized by the development of a rash that starts on the face and spreads to the trunk and limbs. The disease is usually mild in children but can be severe and even life-threatening in adults, pregnant women, and individuals with weakened immune systems.")
    	# st.write("It is spread through direct contact with the rash or through respiratory droplets.")
    elif predicted_label == 'Measles':
    	st.image('measles-1.jpg')
    	st.subheader('Measles:')
    	st.write("Measles is a highly contagious respiratory illness caused by the measles virus. It is characterized by the development of a rash, high fever, and cough. Measles can cause severe and sometimes life-threatening complications, especially in young children, pregnant women, and individuals with weakened immune systems.")
    	# st.write("It is spread through direct contact with the rash or through respiratory droplets.")
    elif predicted_label == 'Monkepox':
    	st.image('Monkeypox_1.jpg')
    	st.write("Monkepox :") 
    	st.write("Monkeypox is a viral illness that is similar to chickenpox. It is caused by the monkeypox virus and is transmitted from animals to humans. The disease is characterized by the development of a rash that starts on the face and spreads to the trunk and limbs. Monkeypox is less severe than smallpox but can still cause serious and sometimes life-threatening complications.")
    	st.write("It is similar to smallpox but less severe.")
    elif predicted_label == 'Smallpox':
    	st.image('smallpox-1.png')
    	st.write("Smallpox :") 
    	st.write("Smallpox is a highly contagious viral illness that was eradicated globally in 1980. The disease is characterized by the development of a rash that starts on the face and spreads to the trunk and limbs. Smallpox was a major public health concern for centuries and was responsible for millions of deaths worldwide before its eradication.")
    	st.write("It is spread through respiratory droplets and direct contact with infected individuals.")
    else:
    	st.write("Unknown disease.")

# Main App
def main():
    pages = {
        "Home": home,
        "Data and Mapping": data_mapping,
         "Diseases Info": Diseases_info,
    }



    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()

if __name__ == "__main__":
    main()
