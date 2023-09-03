# <p align="center"> Speech Emotion Recognition (SER)</p>

- Author: Pranay manikanta Narava
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- GitHub: https://github.com/NPranaya7?tab=repositories 
- LinkedIn: https://www.linkedin.com/in/pranayamanikanta07/ 
- Link to your PowerPoint presentation file
- Link to your  YouTube video 
    
## 1. Background

- What is it about?    
I am excited to explore the fascinating topic of Speech Emotions Recognition. This field involves recognizing human emotions and affective states through speech patterns. It is based on the understanding that our voices often convey our underlying emotions through changes in tone, pitch, and other auditory elements. This phenomenon is not exclusive to humans, as animals like dogs and horses can also comprehend human emotions through our speech patterns. By delving into this subject, we can better understand the intricate nature of human communication and the interplay between our emotions and vocal expressions.

- Why does it matter?    
The significance of emotions in human life cannot be overstated, as they play a pivotal role in facilitating effective communication. Therefore, it would be highly beneficial to create a machine-learning model that can analyze speech and discern an individual's emotional state accurately. Such technology could have numerous practical applications, making it a valuable asset to society.

- Research questions?
    - How accurately can we detect emotions and sentiments from voice recordings?
    - What are the cross-cultural differences in the perception of emotion in audio data?
    - Can emotion analysis from audio data be used in marketing or customer feedback analysis?

## 2. Data 

- Data sources: https://zenodo.org/record/1188976
- Data size: 590.35 MB
- Data shape: 24 Folders each Folder contains 60 Audio (.wav) files
- **What does each row represent?(a patient, a school, a crime, etc.)**
- Data dictionary
  - Columns name
  - Data type
  - Defition
  - Potential values (for categorical valuables, what are the categories?)
- Which variable/column will be your target/label in your ML model?
- Which variables/columns may selected as features/predictors for your ML models?

## 3. Exploratory Data Analysis (EDA)

- Perform data exploration using Jupyter Notebook
- You would focus on the target variable and the selected features and drop all other columns.
- produce summary statistics of key variables
- Create visualizations (I recommend using Plotly)
- Find out if the data require cleansing:
  - missing values?
  - duplicate rows? 
- Find out if the data require splitting, merging, pivoting, etc.
- Find out if you need to bring in other data sources to augment your data.
  - For example, population, socioeconomic data from Census may be helpful.
- For textual data, you will pre-process (normalize, remove stopwords, tokenize) them before you can analyze them in predictive analysis/machine learning.
- Make sure the resulting dataset need to be "tidy":
  - each row represent one observation (ideally one unique entity/subject).
  - each columm represents one unique property of that entity. 

## 4. Model Training 

- What models you will be using for predictive analytics?
- How will you train the models?
  - Train vs test split (80/20, 70/30, etc.)
  - Python packages to be used (scikit-learn, NLTK, spaCy, etc.)
  - The development environments (your laptop, Google CoLab, GitHub CodeSpaces, etc.)
- How will you measure and compare the performance of the models?

## 5. Application of the Trained Models

Develop a web app for people to interact with your trained models. Potential tools for web app development:

- Streamlit (recommended)
- Dash
- Flask

## 6. Conclusion

- Summarize your work and its potetial application
- Point out the limitations of your work
- Lessons learned 
- Talk about future research direction

## 7. References 

List articles, blogs, and websites that you have referenced or used in your project.
