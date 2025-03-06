# fake-news
 - Please see experiments.ipynb, if nothing else has been added
 - Refer to https://rye.astral.sh/ for package management
 - To open the notebook after installing rye, you can first run `rye sync`, then `rye run jupyter lab experiments.ipynb` both from the root of the repo.

# Progress
## Part 1: Data Processing (~1 page)
- [x] Tokenize text
- [x] Remove stopwords and compute the size of the vocabulary. Compute the reduction rate of the vocabulary size after removing stopwords.
- [ ] Apply your data preprocessing pipeline to the 995,000 rows sampled from the FakeNewsCorpus: 995K
- [ ] Task 3: Now try to explore your processed version of the 995K dataset. Make at least three non-trivial observations/discoveries about the data. These observations could be related to outliers, artefacts, or even better: genuinely interesting patterns in the data that could potentially be used for fake-news detection. Examples of simple observations could be how many missing values there are in particular columns - or what the distribution over domains is. Be creative!
- [ ] Describe how you ended up representing the FakeNewsCorpus dataset (for instance with a Pandas dataframe). Argue for why you chose this design.
- [ ] Did you discover any inherent problems with the data while working with it?
- [ ] Report key properties of the data set - for instance through statistics or visualization.

**The exploration can include (but need not be limited to):**
1.    counting the number of URLs in the content
2.    counting the number of dates in the content
3.    counting the number of numeric values in the content
4.    determining the 100 more frequent words that appear in the content
5.    plot the frequency of the 10000 most frequent words (any interesting patterns?)
6.    run the analysis in point 4 and 5 both before and after removing stopwords and applying stemming: do you see any difference?

## Part 2: Simple Logistic Regression Model (~1 page)
- [ ] Task 0: Briefly discuss how you grouped the labels into two groups. Are there any limitations that could arise from the decisions you made when grouping the labels?
- [ ] Task 1: Start by implementing and training a simple logistic regression classifier using a fixed vocabulary of the 10,000 most frequent words extracted from the content field, as the input features. You do not need to apply TF-IDF weighting (expect to achieve an F1 score of ~94% on your test split)
- [ ] Write in your report the performance that you achieve with your implementation of this model, and remember to report any hyper-parameters used for the training process.


# Git cheat sheet
- Basic commands: `git add .`, `git commit -m 'a nice message'` and `git push`
- To sync changes that happened on other branches with their local mirrors, run `git sync`
- To change branch to an existing branch run `git checkout 'branch_name'`
- To create a new branch run `git checkout -b 'new_branch_name'`
- To overwrite the contents of a file, with the version on another branch run `git checkout origin/master 'file_name.ipynb'` (fx if you opened someone elses notebook, do this before you create a pull request)
