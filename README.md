# NLU engine prototype benchmarks
A simple notebook to load several intent classifiers and conditional random fields for entity extraction similar to a voice assistant's NLU engine. These are just prototype engines to perform basic benchmarks (f1 scores) and explain the basic components of an NLU engine. 

##  NLU Intent classifiers
Strangely we couldn't find an example of building a basic intent matcher used in an NLU engine for voice assistants, where utterances (ie user commands/questions) are classified by their intent category. So we decided to make a simple one for everyone out there. 

`NLU_Intent_matching.ipynb` explores the idea of intent matching on utterances and entity extraction. Using the [NLU Evaluation Dataset](https://github.com/xliuhw/NLU-Evaluation-Data) as a bases of data, both a word embedding approach (using Word2Vec) and a TFIDF approach are explored using the following algorithms to train intent classifiers:
* [(Gaussian) Naive Bayes Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
* [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [AdaBoost Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [K-Nearest Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [Support Vector Machine Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

More may be added in the future. 

### Reasons these classifiers were choosen
* Using algorithms that are easy to find in other langauges (not just Python)
* Computationally inexpensive (they could run in real time on a phone, for example)
* Not just inference could be run on low-power device, training could also be performed (TinyML), therefore a user could always add/edit the data to create models that work better for them

### Questions for NLU intent classifiers
* Why does Word2Vec perform much worse than TFIDF?
* Is there any other word embedding based approach that might work better? (how costly is it to run inference on?)
* Would TFIDF out-perform simple bag of words?
* Is f1 a good measure of performance, why?
* How would one fine tune the models and what kind of expectations are there for performance gains?
* How could adding in a entity tagger as a feature boost the intent classification?
* Are there any additional features that could be added beyond TFIDF?

### Entity extraction (Named Entity Recognition)
This notebook uses conditional random fields (CRFs) for entity extraction.

### Questions for NLU CRFs
* What features could be added to improve performance?
* Should the normalized text contain capital letters to add as a feature for the CRF?
* Why do we use just types over type labels with sequences (ie for the beginning and the continouation of the slot) in the label types?

## General Questions
* How do the prototype results compare to more advanced NLU engines?
* Are there any other performance hacks?


# TODO
And there is more to come, whenever we find time. ;)

* Add report discussion of the results in the CSVs
* Compare results to
  * Snips
  * BERT
