# NLP Assignment: Text Classification

##  Mandatory Goals

Train binary classification models for each of the 4 classes of the AG News, and evaluate them onthe test data using accuracy, precision and recall.  Try different choices of ngrams, and learningalgorithms, taking care of proper tuning of model and optimization parameters.  Try your bestmodels and find strong and weak aspects of them.

If  the  data  training  data  is  too  large,  restrict  to  less  examples.   As  much  as  possible,  keep  theoriginal tests, as more testing data yields more reliable evaluations.  Naturally, you can use codein the notebook of the movie reviews.

Describe your experiments as clearly as possible in a report. The report must start with a summary(max 3 pages) that clearly describes what you did, and compiles the main results into appropiate tables or figures that show the relevant aspects of your experiments. Discuss any potential appli-cations for these classifiers, and judge if the classifers you obtained are good enough for a proof ofconcept. Any claim or conclusion should be supported by results or observations you made. Thesame report must have a complete description of what you did after the 3-page summary.  Thiscould be a PDF export of your notebooks, with proper comments in the notebook.  All this said,you’re highly encouraged to be as precise and concise as possible in your writing.


## Optional Goals

We encourage you to run additional experiments that enhance the basic study.   Here are someideas:
- Experiment    with    multicass    methods    that    decide    one    among    the    four    classes.Checkthescikitlearndocumentationonmulticlassmodels(https://scikit-learn.org/stable/modules/multiclass.html):the   models   we   trained   actually   supportmulticlass labels.  Be aware, however, that the convenience functions we wrote to inspectweights assume a binary weight vector.2
- Make learning curves:  train models for increasing amounts of training data, and evaluatethe performance of a model with respect to the size of training. Discuss the results: are thesemethods “data hungry”? Would current performance increase with more data? How costlyis to add more classes to the system?  Note:  learning curves are typically constructing bydoubling the amount of data at each step (i.e. 1, 2, 4, 8, . . . ), you’ll see that for most of thecurve intermediate points are irrelevant.
- Tune the thresholds of your classifiers to obtain very high-recall classifiers, and very high-precision classifiers.  Discuss the trade-off, and what potential applications exists for thesesettings.
- Run a fine analysis on what words and ngrams the model learned (for each class).  Reasonif this makes sense.  For high weights that do not make sense, try to find examples in thetraining data that might explain why the learning algorithm set that weight.
- The “AG News” dataset was compiled many years ago.  Get some news articles from 2021and run the classifiers. See if there’s any noticeable divergence.
- In the lab we could not run the third notebook, which uses Spacy to get linguistic analysisof  textual  data:  proper  tokenization,  syntactic  categories,  named  entities,  . . .   Check  thatnotebook,  and  run  spacy  on  this  data,  in  order  to  analyze  linguistic  aspects  of  it.   Whatnouns, verbs and adjectives are most frequent in each of the categories?  What entities doesSpacy find?  Start with a small sample of documents, and once you are familiar with Spacyand its output, add more documents to your analsys as much as it’s not too demanding foryour computers.
- Try neural networks. Try a CNN! Plenty of tutorials out there!