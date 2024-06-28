# Heart Attack Modeler

### A python script written using Pyspark to create a Machine Learning code modeling heart attack prediction

## Project Description

After taking courses on Spark with Frank Kane and the Sundog Education team, I wanted to try and create a Pyspark program that would use the ML library.

I've always been fascinated by medicine. My father was a Cardiac Anesthesiologist and my first job when taking a "break" from college was working as an assistant perfusionist. I've grown up with my father preaching heart and body health to me. That's why it was such a shock when he recently passed away from a heart attack.

I chose to write this code, knowing there are probably better methods for detecting risk of a heart attack, but the subject is so close to heart (terrible pun) I wanted to try and tackle it myself too.

This code has to .py files. One is just a modeler. I was shaky on making my first model so I kept it separate incase I need to reference it again. The second part is an actual predictor, where you can enter a patients info and it will spit out the 1 or 0. One being if they are at risk of a heart attack and 0 if not. 

## Modeler

This part of the code is pretty simple, I just wanted to see if I could train a model well. I got an accuracy of 81 percent, so I felt comfortable with the model. 

## Predictor
I had a terrible time getting this to work. I had to make sure to downgrade to python 3.11 as 3.12 is not very compatible with Java and/or Pyspark. I had to set my env variables to path to the correct version of python as well.

After getting everything to behave, you can now enter patient data and the predictor spits out a one or zero attached to a predictions table. I'm honestly not sure if it is predicting correctly as I'm not sure how to test it but it was my first crack and machine learning.

## Utilization and Troubleshooting

The modeler should be pretty easy. Make sure you have Java 17 or earlier installed and pathed in environment variables and that you path spark installed and pathed as well. Install the requirements.txt and then the modeler should run just fine on any python IDE. 

The predictor may require more finesse. I had to downgrade to python 3.11, path everything in environmental variables and then make sure my python interpreter was 3.11 and add a os.environ line in my code to point to python 3.11... Even then sometimes it causes issues on restarting. 

Make sure that when you run this you change the os.environ path to your path for python 3.11. Currently there's a place holder.

In my env variables I had to add a path to both my Python and Python\Scripts as well.

After that you run the py file and enter the fields in the terminal.

## Term meanings

Lots of medical jargon in here so some explanation:

- "age": Age
- "sex": Sex (1 for male, 0 for female)
- "cp": Chest pain type (0-3)
- "trtbps": Resting blood pressure
- "chol": Serum cholesterol
- "fbs": Fasting blood sugar (1 if > 120 mg/dl, 0 otherwise)
- "restecg": Resting electrocardiographic results (0-2)
- "thalachh": Maximum heart rate achieved
- "exng": Exercise induced angina (1 for yes, 0 for no)
- "oldpeak": ST depression induced by exercise relative to rest
- "slp": The slope of the peak exercise ST segment (0-2)
- "caa": Number of major vessels (0-3) colored by fluoroscopy
- "thall": Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)

## Future Updates

If I can get the predictor to run more reliably that would be fantastic, I still get java errors sometimes. That and find a larger dataset to train it with

## Credits

A lot of credit needs to go to Frank Cane and his Sundog Education team. Much of this code is just cobbled together from his courses, but doing so taught me how to better utilize it and problem solve.

