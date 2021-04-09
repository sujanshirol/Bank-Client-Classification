# Bank-Client-Classification

The classification of clients applying for loan into bad clients and good clients with respect to the various details regarding the client provided to the bank so the bank could make informative decision to avoid risk of non-repayment of loan and hence reduce liquid damage to the bank.

## Algorithms used:
1. Logistic Regression
2. Gaussian Naive Bayes
3. KNN classifier
4. Decision Tree

## Ensemble techniqu used
1. Random Forest

## Boosting technique used
1. XGBoost

## Other techniques
1. SMOTE data balancing
2. RFE feature selection

# Conclusion
As per the table above XG Boost model gives the best results with an AUC score of 97% and an accuracy score of 92% . However, scores are not only the criteria to decide the best model we also have to take overfitting/underfitting into consideration. XGBoost of course gives best score but is slightly overfitted. Stacked model being 1% less accurate is better balanced than XGBoost and hence, best model is concluded as Stacked model with final estimator as XGBoost.
