# Create CoreML model using your dataset #

**This is an instruction about how to make a basic image classifier CoreML model.**
## Preparation

* Install Turi Create in your computer
* Create your own dataset

### Install Turi Create in your computer

Turi Create is a machine learning frame created by Apple. The frame simplifies the development of custom machine learning models.

* For Linux system, visit [LINUX_INSTALL.md](https://github.com/apple/turicreate/blob/master/LINUX_INSTALL.md)
* For common install issues, vist [INSTALL_ISSUES.md](https://github.com/apple/turicreate/blob/master/INSTALL_ISSUES.md)

It is better to have a virtual environment in your computer to use, install, or build Turi Create.

You can choose Anaconda to get it. The tutorials for Anaconda is here [Tutorials_Anaconda](https://docs.anaconda.com/anaconda/navigator/tutorials/).

Once you enter your virtual environment. It will be easy to install Turi Create. The method for installing Turi Create follows the standard python package installation steps. You can copy the code to your terminal

`pip install -U turicreate`

### Create your own dataset

The next step is to create a dataset.

As it is a sample project, I just got pictures from Google. However, if you want your application is more accurate than other people. The quality of dataset may be important.

* I choose two actors from China, Tony Liang and Aeris Lu.

  Here are two pictures about them.

  Tony Liang 
  
  ![Chaowei Liang](https://raw.githubusercontent.com/Qisheng-Tang/OwnDataset_TuriCreate/master/actordataset/tonyliang/images-2.jpeg)


  Aeris Lu

  ![Aeris Lu](https://raw.githubusercontent.com/Qisheng-Tang/OwnDataset_TuriCreate/master/actordataset/aerislu/images-1.jpeg)

* Collect images

  I just use Google to get 30 pictures for each of them. You can also choose some professional image providers.

  **15 images or more for each label is the basic requirements**

* Assign labels to images

  For machine learning, you should tell the computer the category of images. In this step, you should give images labels.

  For each category, you should create a folder. In this example, I create two folders. The first one is **aerislu**, another one is **tonyliang**. Then assign images to their own folder.

  **The final step** is to create a folder which combines all category folders. In this project, the name is **actordataset**.
  
***
After finishing steps above, you will have a dataset which belongs to you. The next step is to train your model.

## Train your CoreML model
* Code implement
* Test your model

###Code implement

* Open your terminal, create a virtual environment for python.

* Import Turi Create to your file so that you can use it in next steps.
  ```python 
  import turicreate as turi
  ```
After this step, you can use turi instead of turicreate. It will be convenient for you in the future.
  
* Set the path of file and load the images
  ```python
  filepath = "actordataset/"
  data = turi.image_analysis.load_images(filepath)
  ```
In this step, I put actordataset and python file in a same folder, so we just need set the name of dataset folder as the path.
  The **turi.image_analysis.load_images** will import your dataset to the project.

* Give image labels
```python
data["TonyLiangOrAerisLu"] = data["path"].apply(lambda path: "Liang" if "tonyliang" in path else "Lu")
```
  This line of code will read the name of folder as I mentioned above. For images in tonyliang, the program will assign a label as **Liang**. Else, the program will assign a label **Lu** to the rest images.

* Save the data model and view it
```python
data.save("liang_or_lu.sframe")
data.explore()
```
Turi Create provides a method to help you confirm whether you give correct label to each images. You can view label for each image in a visual interface.
* load the sframe file
```python
data = turi.SFrame("liang_or_lu.sframe")
```
* split images to be trained or tested
```python
train_data, test_data = data.random_split(0.8)
```

  In each label, 80% images will be used in training and 20% images will be tested your model. You can choose the proportion by changing the number.

* Create model
```python
model = turi.image_classifier.create(train_data, target="TonyLiangOrAerisLu", max_iterations=30)
```

This functions set max iterations to 30 to get a better result. You can also choose the neural network by using the parameter **model**. Apple now supports 3 neural networks: resnet-50, squeezenet_v1.1, and VisionFeaturePrint_Scene.

 ### Test your model and export to CoreML model

 ```python
 predictions = model.predict(test_data)
 
 
 
 metrics = model.evaluate(test_data)
 
 
 
 print(metrics["accuracy"])
 
 
 
 model.save("qishengActor.model")
 
 
 
 
 
 model.export_coreml("qishengActor.mlmodel")
 ```
 It is easy to understand code above. Perdictions will use test images in the model. Metircs will store the accuracy. You will see the accuracy for you model once you compile you .py file.

 ***

 Following this instructions, you will get to know how to create your CoreML model to use in your application.

 * For documentations about Turi Create, view this link:

https://apple.github.io/turicreate/docs/api/index.html

 * All other materials which may be helpful.

https://github.com/apple/turicreate

  * sample project
  
https://github.com/Qisheng-Tang/OwnDataset_TuriCreate
 
