# Classifying Penguin Species with Machine Learning

**Project Overview:**
- **Objective:** Apply machine learning concepts to classify penguins into species using the Palmer penguins dataset.

**Models Used:**
1. **Neural Network:** Implemented using TensorFlow with a four-layer structure. Optimized for activation functions and loss functions.
2. **Support Vector Machine (SVM):** Utilized SciKit-Learn's SVC, focusing on kernel function optimization.
3. **Logistic Regression:** Also implemented with SciKit-Learn, set to multinomial with optimization for solver and iterations.
4. **k-Nearest Neighbors (kNN):** Used SciKit-Learn’s KNeighborsClassifier with parameter tuning for the number of neighbors.

**Data Preprocessing:**
- The dataset included features like island, bill dimensions, flipper length, body mass, sex, and year of observation. The team conducted preprocessing such as removing samples with missing values, dropping irrelevant features, standardizing measurements, and encoding categorical variables.

**Experimental Approach:**
- **Validation Method:** 10-fold cross-validation was used to ensure consistent training and testing conditions across models.
- **Parameter Tuning:** Extensive testing was done to optimize parameters for each model.

**Results:**
- **Accuracy:** All models achieved high accuracy (over 95%), with SVM reaching 100%. 
- **Best Model:** Despite the highest accuracy from SVM, the team preferred the neural network due to concerns about potential overfitting in other models.

**Conclusion:**
- The project successfully integrated theoretical knowledge and practical application, highlighting the real-world challenges and considerations in model selection and data preprocessing. The team expressed interest in further exploring machine learning in future projects and potential careers.

This summary encapsulates the project's intent, execution, and conclusions while emphasizing the application of multiple machine learning models to solve a classification problem in a real-world-like scenario.

