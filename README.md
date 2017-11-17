# chutya_rating
Uses CV and ML to rate you on a scale of - Kati Chutya lagte rahe ho -> to -> Arey boss?

## Basic Theory
Well, as you might have noticed this, project is for fun -  like most of my projects, but there are quite buzz words involved in making it.
I will use OpenCV, Numpy and scikit-learn to develop a completely automated pipeline that takes a photograph of a person’s face, and rates the photo on how chutya you are.

While applying ML Algo's to a computer vision problem, the high dimensionality of visual data presents a huge challenge. Even a relatively low-res 200×200 image translates to a feature vector of length 40,000. Which is a lot for my tiny machine to work with. Deep learning models like CNN's can work directly with raw images, but they require huge amounts of data to be successful which is agian something I do not have (unless someone wants to make a huge DB of chutiya's :P)

### Feature Extraction
Anyways --> We will need to do some feature extraction instead of feeding the images directly to the machine.
This is we will we taking out the facial landmarks. The following image should clear out any doubt you have on facial landmarks
![Face Landmark](https://www.researchgate.net/profile/Zhiwen_Shao/publication/305388996/figure/fig1/AS:385037640978432@1468811535401/Fig-1-Facial-landmarks-are-divided-into-principal-subset-and-elaborate-subset-And-the.ppm)

We will use [openface](https://github.com/TadasBaltrusaitis/OpenFace) to get the facial landmarks.

### Dataset
Now that we have a way to extract the features, we require a dataset of images. **Which is why i am writing this read me way before publishing the results**

I need your help in creating a dataset, all I need are a few names of people who you think **"SHAKAL SEY CHUTYA LAGTAE HAI"**

### ML bit
Now this is where the things get interestin. We will be using scikit-learn for this.

**Ubuntu**\
`sudo apt-get install python-sklearn`

**Mac OSX**

`pip install -U numpy scipy scikit-learn`

**Windows**
\
`pip install -U scikit-learn`

Once that is done  --> in short, what we want to do here is calculate ratios between all possible pairs of points from our facial landmarks and then use that to train our machine.