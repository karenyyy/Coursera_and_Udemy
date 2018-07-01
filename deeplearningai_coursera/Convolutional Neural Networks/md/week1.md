
## Intro

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/1.png)

If working with very large images, like the images on lower right, shape: 1000x1000x3

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/2.png)

That means X here will be three million dimensional, so in the first hidden layer maybe you have just a 1000 hidden units then the total number of weights that is the matrix W1, which will be (1000, 3m) dimensional matrix. Training a NN with 3m parameters is just infeasible.

__Thus convolution is needed.__

## Edge Detection Example

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/3.png)

> How does Convolution work?
![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/4.png)

### Vertical Edge Detection

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/5.png)

> Why does the vertical edge here appears to be extra thick?

Because the image we are using is too small. If we are using a 1000x1000 image rather than a 6x6 image like we use in the case, we will find that this actually does a pretty good job.

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/7.png)




## Padding

To avoid image shrinking if NN is very deep.

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/8.png)

- __Valid__ and __Same__ convolutions
    - Valid: __No padding__
        - nxn * fxf -> n-(f-1) =  (n -f +1) x (n -f +1)
    - Same: __Pad__ so that output size is the same as the input size
        - (n+2p - (f-1)) x (n+2p - (f-1)) -> (n+2p -f+1) x (n+2p -f+1)
        - n+2p - (f-1) = n , so p=(f-1)/2, thus f is usually odd


## Strided Convolutions

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/%20Convolutional%20Neural%20Networks/images/9.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/%20Convolutional%20Neural%20Networks/images/10.png)




## Convolution Over volumes


__Note: the depth of filters should equal to the depth of the input__

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/11.png)

- If your task is to detect a vertical edge in only one of the color channel, then just assign the filter to that channel, and set the filters of the other channels to be zeros.

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/12.png)

- If multiple filters are used, then:
    - Input : 
        - First layer: height x width x __color channel__
        - Later layer: height x width x __number of filters__ 
    - Filter: 
        - First layer: filter_height x filter_width x __color channel__
        - Later layer: filter_height x filter_width x __number of filters__
    - Output: height x width x __number of filters__
 

## One layer of CNN

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/%20Convolutional%20Neural%20Networks/images/14.png)
![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/%20Convolutional%20Neural%20Networks/images/15.png)


$$\text{ height/width of the next hidden layer} = \frac{n+2p-f}{s} + 1$$


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/%20Convolutional%20Neural%20Networks/images/16.png)




### CNN Example - LeNet 

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/%20Convolutional%20Neural%20Networks/images/18.png)



## Why convolutions? 





- Too many parameters


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/19.png)



-  Parameter sharing

  -  A feature detector (such as a vertical edge detector), that's useful in one part of the image is probably useful in another part of the image.

- Sparsity of connections
  
  - In each layer, each output value depends only on a small number of inputs






