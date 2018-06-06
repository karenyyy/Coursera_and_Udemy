
## Week1

### Train/development(dev)/test datasets

- Notes: 
    - train test split ratio in normal dataset and modern big dataset is quite different.
    - not having test dataset is also ok, just use the dev set as the test set.

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/1.png)

> What if the distribution between training dataset and the validation dataset are mismatched?

__Fix: Make sure the dev and test dataset come from the same distribution!__
![](![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/2.png)

### Bias and Variance (bias-variance trade-off)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/3.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/4.png)

- __Basic recipes for machine learning__
    - High bias (underfitting)
        - bigger network
        - train longer iterations
        - try other neural network architecture approach
    - High variance (overfitting)
        - get more data (data augmentation)
        - regularization
        
![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/5.png)



### Regularization

- For a single layer NN:
$$\min_{w, b} J(w, b)$$

$$\text{ where } J(w,b) = \frac{1}{m} \sum^m_{i=1} L(\hat y^{(i)}, y^{(i)}) + \frac{\lambda}{2m} ||w||_2^2$$

$$\text{ and where },   ||w||_2^2 = \sum_{j=1}^{n_x} w_j^2 = w^Tw$$

- For a multiple-layer NN
$$J(w^{[1]}, b^{[1]}, ..., w^{[L]}, b^{[L]}) = \frac{1}{m}\sum^n_{i=1} L(\hat y ^{(i)},  y^{(i)}) + \frac{\lambda}{2m} \sum^{L}_{l=1} ||w^{[l]}||^2_F$$

$$\text{ where } ||w^{[l]}||^2_F = \sum^{}_{}\sum^{}_{} (w_{ij}^{[l]})^2 $$

So

$$dw^{[l]} = (\text{ from backprop }) + \frac{\lambda}{m} w^{[l]}$$

$$\text{ here from backprop } dw^{[l]} = \frac{1}{m} dz^{[l]}A^{T} $$
$$w^{[l]} = w^{[l]} - \alpha \cdot dw^{[l]}$$


- why can regularization prevent overfitting?

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/6.png)

- One piece of intuition is that, if you crank the regularization lambda to be really big, they will be incentivized to set the weight matrices W to be reasonably close to zero, which is basically zeroing out a lot of the impact of these hidden units
- Vice Versa. If w is really big, then lambda is close to zero;


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/7.png)


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/8.png)

__Note: Put all the intuition aside, zeroing out W in practice is not actually what happens. We should think of it as zeroing out or at least reducing the impact of a lot of the hidden units so you end up with what might feel like a simpler network.__

__The intuition of completely zeroing out a hidden unit isn't quite right. What actually happens is they will still use all the hidden units, but each of them would just have a much smaller effect. But you do end up with a simpler network and as thus less prome to overfitting.__

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/9.png)

- An example exactly why regularization prevent overfitting?

- Use tanh as activation function
- if z is quite small, if z takes on only a smallish range of parameters, __then you are just using the linear regime of the tanh__, only if z is allowed to wander up to larger values or smaller values like so, then the activation function starts to become less linear.
- So, as we know, if lambda is large, w will be small, because they are penalized being large into a cost function, since 

$$z^{[l]} = w^{[l]} a^{[l-1]} +b^{[l]}$$

- if w tends to be very small, then z will also be relatively small, and if z ends up taking relatively small values,  then g(z) will be roughly linear, so if every layer is roughly linear, then the whole network is basically linea, __so not able to fit the very non-linear decision boundaries that allow it to really overfit right to datasets like we saw on the overfitting high variance case__

#### Extra note:
> how to use cost function to debug gradient descent function?

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/10.png)

As we can see, the expected cost decreases monotonically after every elevation of gradient descent.

__Make sure add the regularization part into J!!__

#### Dropout Regularization

- Inverted dropout

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/11.png)

Since the NN value of a is reduced, to make sure the NN value of z remains the same, use inverted dropout, which means increase NN size of a by __a/ = keep.prob__, so that we don't need to add add in an extra scalling parameter at test time, and this inverted dropout technique makes test time when evaluating the NN easier, because we have less of a scaling problem.

__No Dropout at test time! that's just adding noise to the final result__

#### Why does dropout work?

__Intuition: Can't rely on any one feature, so have to spread out weights.__

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/12.png)


#### Other methods of regularization
- Data Augmentation

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/13.png)

- Early Stopping

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/15.png)

### Setting up your optimization problem

#### Normalizing inputs

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/16.png)

__Note: If you normalize your features, your cost function will on average look more symmetric. So when you are running gradient descent on the cost function like the one on the left, might need to use a very small learning rate because GD might need a lot of steps to oscillate back and forth before it finds its way to the minimum.__



#### Vanishing / Exploding gradients

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/17.png)

$$\hat y = 1.5^L \rightarrow gradident \: explode$$

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/18.png)

$$\hat y = 0.5^L \rightarrow gradident \: vanish$$


> Fix: Careful with the weight initialization for deep networks

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/19.png)


__Xavier initialization: if you are using tanh as activation function__

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/20.png)


- another approach
![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/21.png)


> Gradient checking (__use centered gradient(two-sided)__): to make sure your backprop is correct

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/22.png)


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/23.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Improving%20Deep%20Neural%20Networks%3A%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization%20Home/images/24.png)


