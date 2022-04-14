# NLU Engine prototype, benchmarks, and NLU data refinement
simple notebooks to load several intent classifiers and conditional random fields for entity extraction similar to a voice assistant's NLU engine. 

Overview:
* Example of how to build an NLU engine to help developers learn about this topic, including benchmarking
* NLU engine implementation using a simple intent classifier and conditional random fields for entity extraction
* Flows for benchmarking, cleaning and refining NLU datasets

## Learning about NLU engines
We want developers to easily learn how NLU engines work. We want to make it easy to understand the basic components of an NLU engine, and to understand how to use them to perform basic tasks.

###  NLU Intent classifiers
Strangely we couldn't find an example of building a basic intent matcher used in an NLU engine for voice assistants, where utterances (ie user commands/questions) are classified by their intent category. So we decided to make a simple one for everyone out there. 

`NLU_Intent_matching.ipynb` explores the basic idea of intent matching on utterances and entity extraction. It is used mostly for learning purposes. Using the [NLU Evaluation Dataset](https://github.com/xliuhw/NLU-Evaluation-Data) as a bases of data, both a word embedding approach (using Word2Vec) and a TFIDF approach are explored using the following algorithms to train intent classifiers:
* [(Gaussian) Naive Bayes Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
* [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [AdaBoost Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [K-Nearest Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [Support Vector Machine Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

More may be added in the future.


#### Reasons these classifiers were choosen
* Using algorithms that are easy to find in other langauges (not just Python)
* Computationally inexpensive (they could run in real time on a phone, for example)
* Not just inference could be run on low-power device, training could also be performed (TinyML), therefore a user could always add/edit the data to create models that work better for them

#### Questions for NLU intent classifiers
* Why does Word2Vec perform much worse than TFIDF?
* Is there any other word embedding based approach that might work better? (how costly is it to run inference on?)
* Would TFIDF out-perform simple bag of words?
* Is f1 a good measure of performance, why?
* How would one fine tune the models and what kind of expectations are there for performance gains?
* How could adding in a entity tagger as a feature boost the intent classification?
* Are there any additional features that could be added beyond TFIDF?

### Entity extraction (Named Entity Recognition)
This notebook uses conditional random fields (CRFs) for entity extraction.

#### Questions for NLU CRFs
* What features could be added to improve performance?
* Should the normalized text contain capital letters to add as a feature for the CRF?
* Why do we use just types over type labels with sequences (ie for the beginning and the continouation of the slot) in the label types?

### General Questions
* How do the prototype results compare to more advanced NLU engines?
* Are there any other performance hacks?


## Cleaning the NLU dataset
`clean_dataset.ipynb` is a notebook to clean the NLU dataset. The [NLU Evaluation Dataset](https://github.com/xliuhw/NLU-Evaluation-Data), is the only broad voice assistant dataset we could find. However, it does have issues. It is important to clean and refine the dataset. We couldn't find an existing solution that was satisfactory. Therefore, we built a prototype to clean the dataset. You don't have to run this yourself, as the dataset has already been cleaned and is in `data/NLU-Data-Home-Domain-Annotated-All-Cleaned.csv`.


## Building an NLU engine
Based on the teaching example from `NLU_Intent_matching.ipynb`, we built an NLU engine located in `utils/`. The engine is a simple NLU engine that uses intent classifiers and CRFs for entity extraction. By default it uses TFIDF to encode the utterances and CRFs to extract entities. Personally, I am satified with the performance of the intent classifier, however I think the features of the CRFs are worth exploring further. Perhaps Brown clustering could be used to improve the performance of the CRFs. 

TODO: improve the performance of the CRFs, write a report on the feature use to determine which features perform the best and how well they perform.

An example of using the engine is shown in the notebook `example_of_intent_and_entity_classification_with_NLU_engine_class.ipynb`.

I am sure other improvements could be made to the engine, we are open to PRs.


## Macro NLU Data Refinement: work in progress!
The notebook `Macro_NLU_Data_Refinement.ipynb` is a prototype notebook to refine the NLU dataset. It uses the basic NLU engine described above with Logistic Regression as the intent classifier to spot issues. It creates reports on the data and the classifier performance, and it allows a user to go through the different domains (skills, scenarios, whatevere you want to call them) and refine the data using IPYsheets, which are like excel spreadsheets embedded into the notebook. A user can refine both intents and entities. 


## Up next
Once the dataset has been refined, we can use a more advanced NLU engine to train a model. Snips is a good example of a low resource engine while the DistilBERT joint classifier we are building is a good example of a state of the art engine. The models will be publicially released in the future.

TODO:
* Finish the Macro NLU Data Refinement notebook
* Improve the CRF features?
* Finish refining the NLU dataset
* Add report discussion of the results in the CSVs for the basic NLU engine
* Train and export the models as well as a Snips model
* Create DistilBERT joint classifier notebook and export model
* Compare basic results to
  * Snips
  * DistilBERT
* Compress and speed up the DistilBERT model and benchmark
* Create an API and container to easily deploy the DistilBert model (and the other engine?)