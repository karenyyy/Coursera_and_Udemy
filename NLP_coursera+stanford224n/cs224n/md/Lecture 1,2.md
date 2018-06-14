
### Intro to NLP

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/1.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/2.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/3.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/4.png)


### Applications of NLP

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/5.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/6.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/7.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/8.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/9.png)

### word2vec


```python
from nltk.corpus import wordnet as wn
panda = wn.synset("panda.n.01")
print([_ for _ in panda.closure(lambda x: x.hypernyms())]) # closure used as apply?
```

    [Synset('procyonid.n.01'), Synset('carnivore.n.01'), Synset('placental.n.01'), Synset('mammal.n.01'), Synset('vertebrate.n.01'), Synset('chordate.n.01'), Synset('animal.n.01'), Synset('organism.n.01'), Synset('living_thing.n.01'), Synset('whole.n.02'), Synset('object.n.01'), Synset('physical_entity.n.01'), Synset('entity.n.01')]


![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/10.png)

- Since we can not calculate the word similarities using one-hot vectors

Therefore we use another approach, by calculating distributional similarity based on representation. __We can get a lot of value by representing a word by means of its neighbors__


### Basic idea of learning neural neetwork word embeddings

$$p(context\:\mid w_t) = ...$$

which has a loss function, e.g.,

$$J = 1- p(w_{i \text{ where } i \neq t } \mid w_t )$$

where we will:

- look at many positions t in a big language corpus
- keep adjusting the vector representations of words to minimize the loss

### Main idea of Word2vec

- Two algorithms
    - __Skip-grams__ (SG): Predict context words given target (position independent)
    - __Continuous Bag of Words (CBOW)__: Predict target word from bag-of-words context

- Two (moderately efficient) training methods
    - hierarchial softmax
    - negative sampling
    
![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/11.png)


### Details of word2vec

For each word t=1 ... T, predict surrounding words in a window of radius m of every word.

Objective function: Maximize the probability of any context word given the current center word.

$$J'(\theta) = \prod^T_{t=1} \prod _{-m \le j \le m, j \neq 0} p(w_{t+j} \mid w_t; \theta)$$

- Negative Log Likelihood

$$J(\theta) = - \frac{1}{T} \sum^T_{t=1} \sum_{-m \le j \le m, j \neq 0} \log p(w_{t+j} \mid w_t)$$

$$\text{ where } \theta \text{ represents all variables we will optimize } $$

$$p(o \mid c) = \frac{exp (u_o^T v_c)}{\sum^v_{w=1} exp(u_w^T v_c)}, \text{ where o: outside, c: center}$$

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/NLP_coursera%2Bstanford224n/Bayesian_Methods_for_Machine_Learning/images/12.png)


