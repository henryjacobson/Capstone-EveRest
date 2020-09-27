# Capstone-EveRest

For my senior Capstone project, my team and I will be devloping EveRest, a wearable sleep tracking device which provides its users with statistics, analysis, and insights on their sleep behavior, empowering them to ameliorate their sleep health and correct poor sleeping habits. EveRest features both a wristband and a headband device, which operate in tandem to collect high-fidelity sensor data during sleep. A machine learning backend identifies key sleep quality indicators and reports a synthesized sleep score back to the user through a mobile app. We propose a two-layer LSTM-Bayesian-regression architecture for classifying sleep activity. EveRest aims to achieve competitive data integrity compared to medical sleep studies while maintaining the low cost and convenience of commercial wearables.

## My Contributions

My contributions to the project will be in the machine mearning sleep characterization side of things. I developed a plan as to how we will convert raw sensor data to a final sleep score using two separately trained models.

The first model is a long short-term memory (LSTM) neural network architecture which infers sleep characteristics from sensor data. This model will be trained with the Cleveland Family Study (CFS) public dataset. The second model receives outputs from the first, and calculates a compound sleep score for quantitatively describing sleep quality. The second model implements a Bayesian regression model trained from the MMASH (multilevel monitoring of activity and sleep in healthy people) open dataset. The second model is configured to be particularly interpretable to enable further breakdown and analysis of the score.

I will implement the first model, and help to incorporate it into the app along with the second model which will be developed by another student.

## Done so far

### Preprocessing

The raw sensor data needs to go through some preprocessing before it can be plugged straight into a nerual network. To do this I used tools provided by the NSRR who also provided me with the CFS dataset. I used a tool called Luna to extract characterisitc information for each signal in each 5 second interval of a night's sleep. The LSTM will later characterize each interval as either awake or asleep using these intervals as input.

Because there was such a large volume of data, I don't have enough storage available to hold it all at once. Therefore I wrote a script (__preprocess.py__) that will process each polysomnography file individually as followd:

1. Download the .edf (polysomnography) and .xml (annotation) using an NSRR provided gem
1. Run Luna with particular commands on the .edf file to extract relevant information
1. Parse output data for relevant information and store in numpy array
1. Run Luna on .xml for asleep and awake interval labels and store in numpy array
1. Store numpy array in files, delete .edf and .xml, continue to next

After running this script on each polysomnography sample, I have features and labels structured as will be expected for a Tensorflow LSTM. I can them load them and train my network once it has been developed.
