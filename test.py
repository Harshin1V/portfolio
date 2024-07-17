import streamlit as st
import webbrowser


# Initial state for active section
active_section = "About Me"
# Function to display content for each section
def display_section(section_name):
    if section_name == "About Me":
        # col1, col2 = st.columns(2)  # Replace with your profile picture path
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("") 
            st.image("profile-pic.png", width=200)
            st.write("") 
        with col2:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.markdown(" Hey there!!")
            st.markdown("###  I'm Harshini Vutukuri.")
            st.markdown("[View Resume](https://drive.google.com/file/d/1xDGNw2v-vDhq2IUkYELKcuNavqgWLnwF/view?usp=sharing)")    
        
        st.markdown("### About Me👩‍💼")
        st.markdown("Highly motivated and results-oriented recent graduate from `G. Narayanamma Institute of Technology and Science, Hyderabad` from Telangana, India  with a degree in `Computer Science`. Proficient in `Pyhton, ML, DL` and passionate about `AI, ML, DL, NLP, CV, GenAI` . Eager to leverage my skills in a fast-paced startup environment. ")
        st.markdown("### **Skills & Experience👩‍💻**")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("- `Python`")
            st.markdown("- `TensorFlow`")
            st.markdown("- `Pytorch`")
            st.markdown("- `Computer Vision`")
            st.markdown("- `NLP`")
            st.markdown("- `Data Structures`")
            st.markdown("- `LLMs`")
        with col2:
            st.markdown("- `RNNs`")
            st.markdown("- `CNNs`")
            st.markdown("- `OOPs`")
            st.markdown("- `Scikit-learn`")
            st.markdown("- `Linear Algebra`")            
            st.markdown("- `Keras`")
            st.markdown("- `Django`")
        with col3:
            st.markdown("- `Exploratory Data Analysis`")
            st.markdown("- `Matplotlib`")
            st.markdown("- `Seaborn`")
            st.markdown("- `Calculus`")
            st.markdown("- `Probability & Statistics`")        
            st.markdown("- `Streamlit`")
            st.markdown("- `Docker`")
        



        st.markdown("During my `internship at IBM` as a software developer contributed to `IBM Z Firmware Daily Drivers Defect Automation project ` that resulted in complete elimination of manual efforts & improved the Efficiency of Defect creation. ")
        st.markdown("`Fun Fact:` Making computers smarter than ever, but still struggling to understand why humans keep clicking \"I'm not a robot\".")
        
    elif section_name == "Projects":
        st.markdown("## Projects")
        st.markdown("### RNN Classifier with LSTM Trained on Own Dataset (IMDB)")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("- Implementing RNN Classifier on `IMDB dataset` for training a RNN for sentiment classification (Here: It's a `Binary Classification` problem with two labels, positive-1 and negative-0) using `LSTM` (Long Short Term Memory) cells.")
            # st.markdown("- Brief description of Project 1")
            # st.markdown("- Brief description of Project 1")
            st.write(" ")
            st.markdown("[RNN Classifier](https://github.com/Harshin1V/RNN-using-LSTM/tree/main)")
        with col2:
            # st.markdown("[![Alt text for image](Lstm.png)](github_link)")
            st.image("Lstm.png", width=300)

        # 2
        st.write(" ")
        st.markdown("### Object Detection using YOLOv8")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("- Implemented a `YOLOv8`(You Only Look Once) based brain tumor detection system.")
            st.markdown("- Trained and compared the performances of `Nano model and Medium model of YOLOv8`. The training process was optimised with various choice of `Optimizers like Adam, Adamax and RMSprop`. Results can be found in my GitHub Readme.")
            st.write(" ")
            st.markdown("[Object Detection](https://github.com/Harshin1V/Brain-Tumor-Detection-using-YOLOv8)")    
        with col2:
            st.image("yolov8.png", width=300)  # Adjust width as needed
        # 3
        st.write(" ")
        st.markdown("### Deep Generative Modeling GAN & VAE")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("- Implemented a `Deep Convolutional GAN`(Generative Adversarial Network) Trained on `CelebA`(Over `200k images` of celebrities with `40 binary attribute` annotations).")
            st.markdown("- Developed a `VAE`(Variational AutoEncoder) `Latent Space Arithmetic` on `CelebA`. To Compute Average Faces in Latent Space `(Making human faces More or Less Smiling)`.")
            st.write(" ")
            st.markdown("[DCGAN on CelebA](https://github.com/Harshin1V/dcgan-celeba.ipynb)")    
            st.markdown("[VAE celebA Latent Arithmetic](https://github.com/Harshin1V/VAE_celeba_latent-arithmetic)")    
        with col2:
            st.image("gan.jpeg", width=300)  # Adjust width as needed
        # 4
        st.write(" ")
        st.markdown("### BERT for Question Answering")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("- Implemented the `BERT` (Bidirectional Encoder Representations from Transformers) model for `answering questions posed against a context`.")
            st.markdown("- Utilized `BERT base cased model` trained on `SQuAD v2`(Stanford Question Answering Dataset).")
            st.markdown("- Advanced Usage of `Sliding Windows` & `Stride` for Question Answering")
            st.write(" ")
            st.markdown("[BERT for QnA](https://github.com/Harshin1V/BERT-for-Question-Answering)")    
        with col2:
            st.image("bert-img.jpg", width=300)  # Adjust width as needed
        
        # 5-- 
        # st.write(" ")
        # st.markdown("### Object Detection using YOLOv8")
        # col1, col2 = st.columns([1, 1])
        # with col1:
        #     st.markdown("- Brief description of Project 1")
        #     st.markdown("- Brief description of Project 1")
        #     st.markdown("- Brief description of Project 1")
        #     st.write(" ")
        #     st.markdown("[Link to Project](https://github.com/Harshin1V/Brain-Tumor-Detection-using-YOLOv8)")    
        # with col2:
        #     st.image("yolov8.png", width=300)  # Adjust width as needed
        #5

    elif section_name == "Contact Me":
        st.header("Contact Me")

        # Display social media icons (replace with your links)
        st.write("[LinkedIn](https://www.linkedin.com/in/harshini-vutukuri-a4b16321a/)")  # Replace with your LinkedIn URL
        st.write("[GitHub](https://github.com/Harshin1V?tab=repositories)")   # Replace with your Twitter URL
        st.markdown("[View Resume](https://drive.google.com/file/d/1xDGNw2v-vDhq2IUkYELKcuNavqgWLnwF/view?usp=sharing)")    
        
        # Contact form (optional, comment out if not needed)
        contact_form = st.form(key="contact_form")
        name = contact_form.text_input(label="Your Name")
        email = contact_form.text_input(label="Your Email")
        message = contact_form.text_area(label="Message")
        submit_button = contact_form.form_submit_button(label="Submit")

        if submit_button:
            # Add logic to process contact form submission (e.g., send email)
            st.success("Thank you for contacting me! I will get back to you soon.")
            # Replace with your actual contact method (e.g., email sending via library)
            st.write("**Note:** This is a demo, contact information is not sent yet.")

# Layout with columns
col1, col2, col3 = st.columns([1, 1, 1])

# Navigation buttons
with col1:
    if st.button("About Me"):
        active_section = "About Me"
with col2:
    if st.button("Projects"):
        active_section = "Projects"
with col3:
    if st.button("Contact Me"):
        active_section = "Contact Me"

# Display selected section content
display_section(active_section)
