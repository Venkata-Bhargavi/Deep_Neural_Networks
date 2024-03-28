# Deep_Neural_Networks

Code available in branch -- "deep_classifier_develop"

- network_classes.py has fucntions and classes related to network
  
- main.py  data preparation and forward propogation

Input data can be pasted in "/images" directory


  To execute the program:

- `pip install -r requirements.txt`

- Run `python main.py` for running a example



-----------------------------------------------------------------------------------------------

# Network Training

### Objective: To classify Real images vs Artificially generated images (AI images)

Code available in branch -- "Network_Training"

- classes.py : Classes for forward , backward propogation, activation functions and creating dense layers
- main.py : To create training and testing data, iterate over epochs, calculates loss

  Input data can be pasted in "/images" directory, which has sub dorectories - "/authentic" and "/fake" images. In ineterest of memory Images are not uploaded to Git

  
  To execute the program:

- `pip install -r requirements.txt`

- Run `python main.py` to training and classify images

**Hyperparameters**:

- epochs = 100
- learning_rate = 0.001
    
**outputs**: 

![image](https://github.com/Venkata-Bhargavi/Deep_Neural_Networks/assets/114631063/ccfaf809-6009-484a-86c6-76b079a01e80)

