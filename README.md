# fake-news
 - Please see experiments.ipynb, if nothing else has been added
 - Refer to https://rye.astral.sh/ for package management
 - To open the notebook after installing rye, you can first run `rye sync`, then `rye run jupyter lab experiments.ipynb` both from the root of the repo.

# Progress
## Eske
- [ ] Lav IDMA

## Aksel
- [ ] Remove 'political' from label_map
- [ ] Validation of logistic regression trained on full 900,000 rows set
- [ ] Results from logistic regression on full data set and LIAR data set. Table and confusion matrices for both should have same formatting as from the advanced model.


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

## Theo
- [ ] Word frquency analysis of full dataset of first 100 and then 10000 words (and LIAR dataset for part 4) we can also run the word frequency before and after removing stopwords and stemming
- [ ] Count numeric values (maybe) of both data sets
- [ ] Write section in rapport about percentages of labels for the different domains (why adding meta data makes logistic regression unrealistically good)

## William
- [x] Looking good
- [ ] Write part 4
- [ ] Write conclusion

