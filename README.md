# Machine_Learning_Algorithms
Implementing ML algorithms for supervised and unsupervised learning tasks.

The project consists of three parts: (1) supervised learning task, (2) unsupervised learning task, and (3) presentation.


1. Supervised Learning

We consider a classification problem. You will find class data.RData in the module Project on Canvas. It has three variables:

• x: training data covariates 
• y: training data response
• xnew: testing data covariates

We train various classifiers to this dataset. We then summarize all the findings including what classifiers we tried, 
how the paramters were tuned and how the testing error was estiamted etc. in the report. Out of all the classifiers we have 
tried, we report in detail the best classifier such as a desciption, the best set of tuning parameters, the estimated 
testing classification error, etc. 
In addition, we save that best prediction result and estimated testing error using the command
save(ynew,test error,file="###.RData")
where ynew stores the (best) predicted lables and test error stores the estimated testing error. 

2. Unsupervised Learning

You will find another dataset cluster data.RData in the module Project. It has 1000 observations and 784 variables. 
The task is to cluster the observations into an unknown number K of groups. We search the literature and find a meaningful
way to determine K. In the report, we describe the method(s) tried for clustering and choosing K, and state the number of 
clusters we found. 

3. Presentation 

We discuss all the models used in supervised and unsupervised tasks in detail in the presentation.
