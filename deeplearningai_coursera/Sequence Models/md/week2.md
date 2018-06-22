
### Word Representation

__Note: the problem of one-hot is that it can not address to the order of the words in a sentence.__

### Featurized representation: word embedding

#### words get embeddedt as a point into high-dimensional space

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Sequence%20Models/images/26.png)

> How can we determine two words are similar to each other by reading tons of corpus?

> Take the word embedding and apply it to the task, though which we can have a much smaller training set

__For a name entity task, use BRNN__

#### Transfer Learning and Word Embeddings

- learn word embeddings from large text corpus (1-100B words)
    - or download pre-trained embedding online
- transform embedding to new task with smaller training set
- continue to fine tune the word embeddings with new data

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Sequence%20Models/images/27.png)

### Cosine similarity - most commonly used 

$$sim(u,v) = \frac{u^Tv}{||u||_2||v||_2}$$

### Embedding Matrix

When implementing an algorithm to learn a word embedding, what you end up learning is an embedding matrix.

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Sequence%20Models/images/28.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Sequence%20Models/images/29.png)

if the window length is 4, so the words used to predict the next word is the previous 4.

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Sequence%20Models/images/30.png)



## Word2Vec

### Skip-grams

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Sequence%20Models/images/31.png)

$$p(t | c) = \frac{e^{\theta_t^Te_c}}{\sum_{j=1}^{10000} e^{\theta_j^Te_c}}$$

Problem:
- computational speed
    - have to carry out a sum over all 10000 words in the vocab
    
### negative sampling

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Sequence%20Models/images/32.png)

> How to choose the negative examples?

- A heuristic approach:

$$P(w_i) = \frac{f(w_i)^{\frac{3}{4}}}{\sum_{j=1}^{10000} f(w_j)^{\frac{3}{4}}}$$


### GloVe Algorithm

#### global vectors for word repersentation

$$X_{ij} = \text{ times i } \rightarrow t \text{ appears in context of } j \rightarrow c$$

$$X_{ij} = X_{ji} \rightarrow \text{ captures the counts of i and j appear together}$$

what the GloVe model does is that it optimize the following:

$$\min (\theta_i^T e_j - \log X_{ij})^2 \rightarrow \min (\theta_t^T e_c - \log X_{tc})^2, \text{ where t: target word; c: context word }$$

$$\text{How related are words t and c are measured by how often they occur with each other}$$

> How to solve theta and e?

$$\text{solve for the parameters } \theta \text{ and } e \text{ using gradient descent to minimize the sum}$$

$$\text{weighting term: } f(X_{ij}) $$

$$\min \sum_{i=1}^{10000} \sum_{j=1}^{10000} f(X_{ij})(\theta_i^T e_j +b_i +b_j'  \log X_{ij})^2$$

### Sentiment Analysis

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Sequence%20Models/images/33.png)

![](https://raw.githubusercontent.com/karenyyy/Coursera_and_Udemy/master/deeplearningai_coursera/Sequence%20Models/images/34.png)

