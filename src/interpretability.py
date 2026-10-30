import numpy as np
import pandas as pd

def compute_local_attribution(forest, X_sample, feature_names):
    """
    Computes exact local feature attribution (contributions) for a random forest classifier
    based on decision paths.
    
    Parameters:
    -----------
    forest : RandomForestClassifier
        The trained scikit-learn random forest model.
    X_sample : array-like or DataFrame
        A single sample to explain. Shape (n_features,) or (1, n_features).
    feature_names : list
        List of feature names matching the columns of X_sample.
        
    Returns:
    --------
    dict
        A dictionary mapping feature_name -> contribution to the probability of class 1 (Critical).
        And 'bias' which is the average root node probability of class 1 across all trees.
    """
    # Standardize input to 2D numpy array
    if isinstance(X_sample, pd.DataFrame):
        X_arr = X_sample.values
    elif isinstance(X_sample, pd.Series):
        X_arr = X_sample.values.reshape(1, -1)
    else:
        X_arr = np.array(X_sample)
        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1, -1)
            
    n_features = len(feature_names)
    contributions = np.zeros(n_features)
    biases = []
    
    # Iterate over all trees in the forest
    for tree in forest.estimators_:
        node_indicator = tree.decision_path(X_arr)
        # Apply returns the leaf node index for the sample
        leave_id = tree.apply(X_arr)[0]
        
        # Get values (sample counts per class) at all nodes in this tree
        # tree_.value shape is (n_nodes, 1, n_classes) or (n_nodes, n_classes)
        value = tree.tree_.value
        if len(value.shape) == 3:
            value = value.squeeze(axis=1)
            
        # Compute probabilities of class 1 (Critical) at each node
        node_sums = value.sum(axis=1, keepdims=True)
        # Avoid division by zero
        node_sums[node_sums == 0] = 1.0
        probs = value / node_sums
        
        # We are predicting probability of class 1 (index 1)
        prob_class_1 = probs[:, 1]
        
        # Get decision path node indices for the first (and only) sample
        path = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        
        # Record the root node probability as bias
        root_node = path[0]
        biases.append(prob_class_1[root_node])
        
        # Accumulate contributions along the path
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i+1]
            feature_idx = tree.tree_.feature[parent]
            if feature_idx >= 0:  # Valid split feature (not leaf)
                delta = prob_class_1[child] - prob_class_1[parent]
                contributions[feature_idx] += delta
                
    # Average contributions and bias across all estimators
    n_trees = len(forest.estimators_)
    mean_contributions = contributions / n_trees
    mean_bias = float(np.mean(biases))
    
    explanation = {feature_names[i]: float(mean_contributions[i]) for i in range(n_features)}
    
    # Sort contributions by absolute value (descending)
    sorted_explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True))
    
    return {
        "bias": mean_bias,
        "contributions": sorted_explanation
    }
