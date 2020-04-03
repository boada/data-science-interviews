## Statistical Inference (16 questions)

#### 1. In an A/B test, how can you check if assignment to the various buckets was truly random?

#### 2. What might be the benefits of running an A/A test, where you have two buckets who are exposed to the exact same product?

#### 3. What would be the hazards of letting users sneak a peek at the other bucket in an A/B test?

#### 4. What would be some issues if blogs decide to cover one of your experimental groups?

#### 5. How would you conduct an A/B test on an opt-in feature? 

#### 6. How would you run an A/B test for many variants, say 20 or more?

#### 7. How would you run an A/B test if the observations are extremely right-skewed?

#### 8. I have two different experiments that both change the sign-up button to my website. I want to test them at the same time. What kinds of things should I keep in mind?

#### 9. What is a p-value? What is the difference between type-1 and type-2 error?

#### 10. You are AirBnB and you want to test the hypothesis that a greater number of photographs increases the chances that a buyer selects the listing. How would you test this hypothesis?

#### 11. How would you design an experiment to determine the impact of latency on user engagement?

#### 12. What is maximum likelihood estimation? Could there be any case where it doesn’t exist?

#### 13. What is the liklelihood ratio test?

The likelihood ratio test is for comparing nested models of different complexity. It assesses whether the better fit of the more complex model is statistically significant. It does this in the following way: 

Let the likelihood L(M) of a model M be P(Data|M). 

Then given our models M1 and M2, the *likelihood ratio* LR is: 

LR = L(M1) / L(m2). 

Let 
Delta = ln(LR^2) = 2*(ln(L(M1) - ln(L(M2))).

If the simpler model is the "true" model, then Delta follows a chi-square distribution, with degrees of freedom equal to the number of extra parameters in the more complicated models. So, setting as our null hypothesis that the simpler model is the true model, we can test whether the difference between the expected value of Delta under that hypothesis (given by the chi-square distribution) and our observed value is statistically significant. 


https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.2041-210X.2010.00063.x

https://astrostatistics.psu.edu/su08/lecturenotes/rao_model08.pdf

https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqhow-are-the-likelihood-ratio-wald-and-lagrange-multiplier-score-tests-different-andor-similar/


#### 14. How can you test whether your respone variable (or for that matter any parameter) follows a given distribution? (Not just Gaussian)?

#### 15. What’s the difference between a MAP, MOM, MLE estima\- tor? In which cases would you want to use each?

#### 16. What is a confidence interval and how do you interpret it?

#### 17. What is unbiasedness as a property of an estimator? Is this always a desirable property when performing inference? What about in data analysis or predictive modeling?

#### 18. What is Selection Bias?
  

