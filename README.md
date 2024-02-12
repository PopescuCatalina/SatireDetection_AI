In this solution, the focus is on satire detection in two datasets, FreSaDa , a French satire dataset, for multi-domain satire detection, which is composed of 11,570 articles from the news domain, and the SaRoCo dataset, which is a dataset for detecting satire in Romanian News containing 55,608 news articles from multiple real and satirical news sources, of which 27,980 are regular and 27,628 satirical news. It provides data in csv format, in three files following train/validation/test splits.
In the next part, three classification methods CamemBERT, Ro-BERT tuned Romanian BERT) and Bert will be taken to compare them in detecting satire on original and translated texts. The French dataset will be translated to English using the language transformers because
there are more resources in English and Bert is much better prepared and we will proceed accordingly with the second. In the final part, a method of applying the logistic regression model on the concatenation of the weights of both data sets was tried.

![Screenshot 2024-02-12 104925](https://github.com/PopescuCatalina/SatireDetection_AI/assets/43966104/9ea6ec80-cde8-4802-9f69-677654fdccbd)

![Screenshot 2024-02-12 104946](https://github.com/PopescuCatalina/SatireDetection_AI/assets/43966104/97d8a929-ac57-4c61-af8b-fef42f957b76)


![Screenshot 2024-02-12 104702](https://github.com/PopescuCatalina/SatireDetection_AI/assets/43966104/ff81652e-d8b6-4141-b9bc-ceaeac4e1a7c)


![Screenshot 2024-02-12 104605](https://github.com/PopescuCatalina/SatireDetection_AI/assets/43966104/c02765c5-531b-4349-8d1f-1d66c39ad161)


![Screenshot 2024-02-12 104340](https://github.com/PopescuCatalina/SatireDetection_AI/assets/43966104/575e31fa-ac7f-4aaa-8dd6-918c16aef88f)


![Screenshot 2024-02-12 104507](https://github.com/PopescuCatalina/SatireDetection_AI/assets/43966104/9086c81d-b062-40ff-9c57-bc23bc4b6a4f)

In a final conclusion on the decisions either to translate the texts into English, to add
weights randomly or to pass the final layers through a classification algorithm, I came to a
clear conclusion. Of course, the English translation brought a plus of suggestions in
understanding the processing of the text and using the BERT model on these texts, but the
models pre-trained on texts in Romanian or French have become much more strongly pretrained than in the past, so the need to translate into English in order to obtain a good
accuracy is no longer so necessary.
If we talk instead about the two experiments of adding weight to the weight randomly
or through a classification model, here the method of the classic binary classification model
has won.
As we could see in the above description of the results, the highest accuracy came
from the binary classification model, which was a result to be expected, considering that in
the first variant we chose the weights according to the individual performance of each model,
which was not a solution to come from giving the weights with information related to the data
sequences from the previous training processes.
