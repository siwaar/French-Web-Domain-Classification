# French Web Domain Classification
Authors : ABBES Siwar, ABED Dina, Karray Elyess

## Introduction
The goal of this challenge is to solve a web domain classification problem. Text based methods and Graph based methods on
web pages document for classification tasks are ofmany studies.
In this project, we implement different prepossessing strategies to deal with our data and we explore different methods of machine learning and deep learning to solve a web domain classification problem. We present all the methods deployed and the
best scored method.

## Dataset
The dataset of French domains was generated from a large crawl of the French web that was performed
by the DaSciM team. You are given the following files:
1. edgelist.txt: a subgraph of the French web graph. It has 28, 002 vertices and 319, 498 weighted,
directed edges. Nodes correspond to domain ids and there is an edge between two nodes if there
is a hyperlink from at least one page of the source domain to at least one page of the target domain.
2. text directory: for each domain, a .txt file containing the text of all the pages of the domain. The
text was extracted from the HTML source code of the pages.
3. train.csv: 2, 125 labeled domain ids. One domain id and category per row. The list of categories is
shown in Table 1.
4. test.csv: 560 domain ids the category of which is to be predicted. One domain id per row.
5. graph baseline.csv: output of the provided graph baseline. Submissions have to follow this exact
format.
