import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np;
image = Image.open('reuse.png')
st.image(image)
st.title("Waste Segmentation and Disposing Recommender System")
uploaded_image = st.file_uploader("Upload image of waste here", type=["jpg", "png", "jpeg"])
import tensorflow as tf
st.image(uploaded_image)

# Display the image in your Streamlit app

# Load the saved model
model = tf.keras.models.load_model('classifyWaste.h5')
from tensorflow.keras.preprocessing import image
output_class=["batteries","clothes","e-waste","glass","light bulbs","metal","organic","paper","plastic"]
def waste_prediction(new_image):
    test_image=image.load_img(new_image,target_size=(224,224))
    plt.imshow(test_image)
    plt.show()
    
    test_image=image.img_to_array(test_image)/255
    test_image=np.expand_dims(test_image,axis=0)
    predicted_array=model.predict(test_image)
    predicted_value=output_class[np.argmax(predicted_array)]
    predicted_accuracy=round(np.max(predicted_array)*100,2)
    st.header(f"**Your Waste Material is {predicted_value} with {predicted_accuracy}% accuracy**")
    if predicted_value=="batteries":
        st.write("Batteries, both single-use and rechargeable, should be recycled at specialized collection points or recycling centers. Many electronic stores and recycling facilities accept batteries for safe disposal.")
    if predicted_value=="clothes":
         st.write("Gently used clothes can be donated to charities or sold online. Worn-out clothing can be recycled at textile recycling centers, repurposed into cleaning rags, or even composted if they are natural fibers.")
    if predicted_value=="e-waste":
        st.write("E-waste, such as old computers, phones, and electronics, should be recycled through e-waste recycling programs or designated drop-off locations. Avoid disposing of electronics in regular trash.")
        
    if predicted_value=="glass":
        st.write("Glass containers, like bottles and jars, should be rinsed and placed in designated glass recycling bins. Recycling glass conserves resources and reduces landfill waste.")
    if predicted_value=="light bulbs":
       st.write("Compact fluorescent lamps (CFLs) and fluorescent tubes contain small amounts of mercury. These should be taken to hazardous waste facilities for proper disposal. LED bulbs are safer and can be recycled.")
    if predicted_value=="metal":
        st.write("Metal items, including aluminum cans and steel products, are highly recyclable. Place them in your recycling bin for collection or take them to a scrap yard.")
    if predicted_value=="organic":
        st.write("Organic waste, like food scraps and yard trimmings, can be composted in home compost bins or taken to municipal composting facilities. Composting reduces landfill waste and enriches soil.")
    if predicted_value=="paper":
        st.write("Most paper products, such as newspapers, cardboard, and office paper, can be recycled. Use designated paper recycling bins or collection services to dispose of paper waste.")
    if predicted_value=="plastic":
        st.write("Recycle plastic containers with the appropriate recycling code (e.g., PET, HDPE). Check your local recycling guidelines for specific plastic recycling instructions. Avoid single-use plastics when possible.")
    
   
   
if uploaded_image is not None:
    waste_prediction(uploaded_image)