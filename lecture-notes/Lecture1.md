# STATS 315: Lecture 1

## Topics
* Deep Learning and the rise of it
    * Example: DALL-E 2 (OpenAI)
        * Has made for some great Twitter posts
    * GitHub Copilot (GitHub, OpenAI)
    * GPT-3 (OpenAI)
    * Minera (Google)
        * Used to solve math problems, trained on IMO Problems


### What is Deep Learning?
* Act of extracting patterns from data using neural networks
* AI: Any technique that enables computers to mimic human behavior
* ML: Ability to learn without explicitly being programmed

### Goals for Today
* Course Logistics
* What is deep learning? (Core constructs, basic structures, etc.)

## Course Information

### What is this course about?
* Deep Learning:
* Machine Learning:
* Statistics (Stats): Science of learning from data. ML and stats have been coming increasingly close over the past 2 decades
* Data Science (DS): Emerging discipline that seeks to marry statistical thinking with computational thinking to solve difficult real world problems.
* Tenserflow (TF)/Keras:

### Prereqs
* Calc: Derivatives, gradients, chain rule
* Prob/Stats: Random Variablses, Expecation, Linear Regression
* Programming: Variables, Data Structures, Loops, Functions, Classes, Objects
* NO ASSUMPTION ON Machine Learning or Python
* Can feel intense due to covering necessary Stats/ML and Python materials to get you started

### Books
* Will not need to buy ANY textbooks, will be available online or via Canvas!

### Grading
* Refer to syllabus online, will be curved if necessary but minimums guaranteed.

### Languages
* Mainly demo in Python (Tensorflow)
* Can use other languages if desired

### HW Assignments
* Homework will not be an execution of what is covered in lecture
    * You will learn new things, especially on the practice of data analysis
* Quizzes are different! We'll talk about those shortly

### Reading Assignments
* Tentative course schedule online
* Each lecture has corresponding reading assignments
* Material not available online will be on Canvas
* They will be needed to do well on JisTTs and in the overall class

### Academic Integrity
* Can discuss HWs but NOT quizzes
* All submitted work, including code, must be your own (otherwise gg no re if you are reported)
* When in doubt, ask!

### Labs
* Labs will be posted online
* Attendance is NOT mandatory

### Course Outline (the fun stuff)
* Python, Numpy, TF2, Keras, Jupyter Colab
* Simple models that are precursors to DL; Linear Regression
* Fully connected multilayer Neural Networks (NNs)
* Convolutional neural networks and vision
* Sequence models and langauge


## Introduction to Deep Learning (actual content)

### Beginnings of AI (1940s-1950s)
* Alan Turing's seminal papers
    * Intelligent Machinery
    * Compiling Machinery and Intelligence


### Symbolic AI
* Relied on hand-crafted rules for manipulating knowledge stored in databases
    * Cyc had over 24 million rules for about over 1 million object in its ontology in 2017
* Dominant apporach in the 1950's to the 1980's
* Reached its peak with the expert systems boom of the 1980's
* Tasks that are easy for people to do but hard to formalized proved challenging
    * Recognizing spoken words or faces in images
* Classical Programming: Take in Rules and Data, output answers
* ML: Take in Data and Answers, output rules


### Machine Learning
* An ML system is trained instead of being explicitly programmed
* Started in the 1990's
* Has had an explosive growth driven by faster hardware and larger datasets
* Example: wake word derection in voice assistants (Alexa, Siri, Google Assisstant)

### Wake Word Detection
* Do not know how to program a computer to recognize a word


### Representations
* ML Model transforms input data (audio) into meaningful output (is the wake word in it)
* Different representations are useful for different tasks: consider images in RGB (red-green-blue) vs HSV (hue-saturation-value) formats
    * "Select all red pixels" easier in RGB
    * "Make image less saturated" easier in HSV
* We can make tasks easier by choosing better representations

### Deep Learning
* Learn successive layers of increasingly meaninful representations
* Enables computer to build concepts out of simpler concepts
* Modern DL involves rens and sometimes hundreds of layers
* All of the parameters in these layers are learned from data

### Why Now?
* Lots of interest has arisen in AI, ML, and DL
* Couple of AI winters during the past 60 years
* NN Research had 3 waves
    * Cybernetics (1940's - 1960's)
    * Connectionism (1980's - 1990's)
    * Deep Learning (2006 - now)
* Prior to 2010: Algorithms too computationally costly for the hardware at the time
* 2006: Geoff Hinton showed a type of NN called "deep belief network" could be efficiently trained
* Wave of NNs research came, popularized "deep learning" as a term
* Researchers able to train deeper NNs than before
* Attention focused on theoretical importance of depth

### Scale Drives Deep Learning Progress
* Data, Computation, Algorithms

### Short term hype vs long term hype
* Expectations for what the field will be able to achieve in the enxt decade tend to run much higher than what will likely be possible
* Believable human dialogue systems, human-level machine translation across arbitrary languages, human-level natural language understanding may remain elusive for a long time.
* May be seeing the 3rd cycle unfolding, with hype and disappointment, still in optimism. Will there be a 3rd AI winter?
* Long term hype more believable
    * AI helps humanity as a whole, breakthrough discoveries across all fields
    * Capitalism blah blah blah

### Ethical Issues
* Energy consumption, sustainability, climate change
* Social, economic, political inequality
* Unemplyment
* Autonomous weapons
* Suverillance and loss of privacy