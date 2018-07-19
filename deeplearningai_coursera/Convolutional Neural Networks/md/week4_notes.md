
## Face Recognition

### face verification vs. face recognition

Verification
- Input image, name/ID
- Output whether the input image is that of the claimed person

Recognition
- Has a database of K persons
- Get an input image
- Output ID if the image is any of the K persons (or 'not recognized')





### One-shot Learning


Learning from one example to recognize the person again

### Learning a 'similarity function

$$d(img1, img2) = \text{degree of difference between images}$$


$$\text{If } d(img1, im2) \le x, \Rightarrow \text{ same image }$$



### Siamese Network



### Triplet Loss


- A: Anchor picture
- P: Positive picture
- N: Negative picture

$$\text{Goal: }\frac{||f(A） － f(p)||^2}{d(A,p)} \le \frac{||f(A） － f(N)||^2}{d(A,N)}$$




Given 3 images, A, P, N

$$L(A, P, N) = \max||f(A) - f(P)||^2-||f(A) - f(N)||^2 +\alpha, 0)$$


$$J = \sum^m_{i=1} L(A^{(i)}, P^{(i)}, N^{(i)})$$


use __Gradient Descent__




### Turn the last the fc layer of face verification into a binary classification problem



![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/46.png)



$$\hat y = \sigma \left ( \sum^{128}_{k=1} w_i |f(x^{(i)})_k - f(x^{(j)})_k| + b\right) $$








## Neural Style Transfer



what are the deeper layers of CNN doing?

__Each hidden unit pinpointing one feature of the input picture:__


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/47.png)


__Each 3x3 grid cell represents one feature captured by the CNN__


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/48.png)


As the layer gets deeper, it captures more complex features.


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/49.png)


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/50.png)


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/51.png)


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/52.png)


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/53.png)






##  Cost Function

$$J(G) = \alpha J_{content}(C, G) +\beta J_{style}(S, G)$$


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/54.png)



#### Find the generated image G


- Initiate G randomly

  - G: 100x100x3

- Use gradient descent to minimize $J(G)$


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/55.png)
















### Content cost function


- use hidden layer $l$ to compute content cost
- use __pre-trained__ ConvNet (VGG)
- let $a^{[l](C)}$ and $a^{[l](G)}$ be the activation of layer $l$ on the images
  - if $a^{[l](C)}$ and $a^{[l](G)}$ are similar, both images have similar content

$$J_{content} (C, G) = \frac{1}{2}||a^{[l](C)} - a^{[l](G)}||^2$$


### Style cost function


- use layer $l$'s activation to measure 'style'
- define style as correlation between activations across channels


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/56.png)




![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Convolutional%20Neural%20Networks/images/57.png)



__Why does the correlations between activations across channels represent 'style'???__

example:

- red channel:  the 2nd neuron
- yellow channel: the 4th neuron


__if these 2 channels are highly correlated, then that means whatever part of image has this type of subtle vertical texture (2nd neuron), that part of the image will probably have the orange-ish tint (4th neuron)__


#### And the degree of correlation provides one way of measuring how  often these different high level features such direction pattern or texture, how often they occur and how often they occur together in different parts of an image.


- then we can measure the degree to which in the generated image, one channel is correlated ot uncorrelated with the another channel, that  tells us how similar is the style of the generated image to the style of the input style image

### Style Matrix

$$\text{Let } a^{[L]}_{i,j,k} = \text{ activation at } (i,j,k)$$

$$G^{[l]} = n_c^{[l]} \text{ x } n_c^{[l]}$$



$$G_{kk'}^{[l](S)} = \sum_{i=1}^{n_{h}^{[l]}} \sum_{j=1}^{n_w^{[l]}} a^{[L](S)}_{i,j,k} a^{[L](S)}_{i,j,k'} , \text{ where } k, k' = 1,..., n_c^{[l]}$$



$$G_{kk'}^{[l](G)} = \sum_{i=1}^{n_{h}^{[l]}} \sum_{j=1}^{n_w^{[l]}} a^{[L](G)}_{i,j,k} a^{[L](G)}_{i,j,k'} , \text{ where } k, k' = 1,..., n_c^{[l]}$$



__What G is doing is summing over the different positions that the image over the height and width and just multiplying the activations together of the channels k and k'__



$$J_{style}^{[l]}(S, G) = \frac{1}{(2n_h^{[l]}n_w^{[l]}n_c^{[l]})^2} ||G_{kk'}^{[l](S)}  - G_{kk'}^{[l](G)} ||^2_F$$


$$=\frac{1}{(2n_h^{[l]}n_w^{[l]}n_c^{[l]})^2} \sum_k \sum_{k'} (G_{kk'}^{[l](S)} - G_{kk'}^{[l](G)} )^2$$


It turns out that you get more visually pleasing results if you use the style cost function __from multiple different layers__.


#### Overall style cost function

$$J_{style}(S, G) = \sum_l \lambda^{[l]} J_{style}^{[l]}(S, G)$$


__It allows you to use different layers in a neural network, including early ones which measures relatively simpler low level features like edges as well as later layers which measure high level features and cause a NN to take both high and low level correlations into account when computing style.__


















