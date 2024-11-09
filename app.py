import streamlit as st
import webbrowser

col1, col2 = st.columns([1, 1])
with col1:
    st.write("") 
    st.image("profile-pic.png", width=200)
    st.write("") 
with col2:
    st.markdown("### </u>Harshini Vutukuri</u> ", unsafe_allow_html=True)
    # st.markdown("I'm Harshini Vutukuri.Highly motivated and results-oriented recent graduate from `G. Narayanamma Institute of Technology and Science, Hyderabad` from Telangana, India  with a degree in `Computer Science`. Proficient in `Pyhton, ML, DL` and passionate about `AI, ML, DL, NLP, CV, GenAI` . Eager to leverage my skills in a fast-paced startup environment. ")
    st.markdown("`human = True` with experience in software development , skilled in `Python`, machine learning `ML`, and deep learning `DL`. Strong passion for `NLP`, and computer vision`CV`, ready to make an impact in fast-paced, innovative environments.")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("[Resume](https://drive.google.com/file/d/1GVkPIUYlIvuxJpSIvV1cIk6JnbyfydSI/view?usp=sharing)")
    with col2:
        st.markdown("[LinkedIn](https://www.linkedin.com/in/harshini-vutukuri-a4b16321a/)")

st.markdown("## Projects🏅")
# 1

st.markdown("### Detecting bullying Tweets `LSTM & BERT` [GitHub](https://github.com/Harshin1V/Detecting-bullying-Tweets-PyTorch-LSTM-BERT/blob/main/detecting-bullying-tweets-pytorch-lstm-bert.ipynb)")
st.markdown("- **Objective**: Analyse tweets to <u>Detect cyberbullying,</u> categorizing bullying content by themes like religion, age, race, and gender, using <u>Sentiment analysis.</u>", unsafe_allow_html=True)
st.markdown("- **Approach:** Implement and compare models, starting with a <u>Naive Bayes baseline,</u> then an <u>LSTM with Attention,</u> and finally <u>BERT,</u> utilizing PyTorch for model training and evaluation.", unsafe_allow_html=True)
st.markdown("- **Results:** The <u>BERT classifier significantly outperforms the LSTM model,</u> achieving an overall <u>accuracy of around 95%</u> and <u>F1 scores above 95%.</u>", unsafe_allow_html=True)
st.write(" ")

# 2
st.markdown("### Detect Fake Tasks using `CGAN` [GitHub](https://github.com/Harshin1V/Detect-Fake-Tasks-using-CGAN/tree/main)")
st.markdown("- **Objective:** <u>Train a Discriminator</u> to differentiate between real and fake data due to mobile crowdsourcing concerns, where fake data affects model performance.", unsafe_allow_html=True)
st.markdown("- **Methodology:** Applied <u>AdaBoost and Random Forest</u> on real data, then added synthetic data. A <u>two-level classification</u> was used, where the <u>Discriminator filtered fake data,</u> followed by AdaBoost/Random Forest for classifying real tasks.", unsafe_allow_html=True)
st.markdown("- **Results:** Without the Discriminator, fake data reduced accuracies: <u>AdaBoost from 0.92 to 0.575</u>; <u>Random Forest from 0.993 to 0.590</u>. After filtering, accuracies rose again to <u>0.926 for AdaBoost</u> and <u>0.993 for Random Forest,</u> restoring near-original performance.", unsafe_allow_html=True)
st.write(" ")

# 3
st.markdown("### Lottery Ticket Hypothesis [GitHub](https://github.com/Harshin1V/Lottery-Ticket-Hypothesis/tree/main)")
st.markdown("- **Objective**: Implemented the methodology of: <u>The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks,</u> which posits that sparse subnetworks <u>Winning tickets</u> within dense neural networks can achieve <u>Comparable accuracy when trained independently.</u>", unsafe_allow_html=True)
st.markdown("- **Method**: Employ <u>Iterative Pruning</u> to identify effective subnetworks while comparing the impact of maintaining <u>Initial weights</u> versus <u>Random reinitialization.</u>", unsafe_allow_html=True)
st.markdown("- **Results**: These winning tickets allow for efficient <u>Training</u> and <u>Storage,</u> as subnetworks retain high <u>Accuracy</u> with only <u>10-20%</u> of the original network's size.", unsafe_allow_html=True)
st.write(" ")

# 4
st.markdown("### Image Captioning using `CNNs+LSTMs` [GitHub](https://github.com/Harshin1V/Image-Captioning-using-CNNs-LSTMs/tree/main)")
st.markdown("- **Model Overview**: The project implements image captioning using a <u>CNN (DenseNet201)</u> for <u>Feature extraction</u> and an <u>LSTM</u> for <u>Generating captions.</u> Text preprocessing includes tokenization, encoding, and embedding to form input for the LSTM.", unsafe_allow_html=True)
st.markdown("- **Training**: The model is trained with batch-wise data generation, <u>Combining image embeddings</u> with <u>Text embeddings</u> to <u>Predict captions.</u> Adjustments to the architecture enhance performance.", unsafe_allow_html=True)
st.markdown("- **Inference and Evaluation**: The model generates captions during inference by <u>Feeding image features</u> and <u>Word embeddings</u> into the LSTM.", unsafe_allow_html=True)
st.write(" ")

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
    st.markdown("- `Statistics`")
with col3:
    st.markdown("- `Exploratory Data Analysis`")
    st.markdown("- `Matplotlib`")
    st.markdown("- `Seaborn`")
    st.markdown("- `Calculus`")
    st.markdown("- `Probability`")        
    st.markdown("- `Streamlit`")
    st.markdown("- `SQL`")
st.markdown("**Work Experience:** Worked towards Improving softwares in `z-Firmware Delivery` for `IBM Infrastructure` by Developing `End-to-End Automation system` for Defect Creation and Handling  using `python.`")
