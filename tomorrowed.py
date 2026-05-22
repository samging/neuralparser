for epoch in range(epochs):
    # Reset our gradient accumulators to zero for the new epoch
    total_loss = 0
    grad_W1, grad_B1 = 0, 0
    grad_W2, grad_B2 = 0, 0
    grad_intercept = 0
    
    # 1. FORWARD PASS & GRADIENT CALCULATION (Loop through all networks)
    for net in meta_dataset:
        # Extract shapes
        X1, Y1, X2, Y2, R = net["X1"], net["Y1"], net["X2"], net["Y2"], net["R"]
        A = net["true_accuracy"]
        
        # Calculate prediction using current global state
        A_hat = (X1 * Y1 * W[0]) + (Y1 * B[0]) + R * ((X2 * Y2 * W[1]) + (Y2 * B[1])) + global_intercept
        
        # Accumulate total loss
        total_loss += 0.5 * ((A_hat - A) ** 2)
        
        # Calculate the base error signal
        error_signal = (A_hat - A)
        
        # Accumulate the gradients for this specific network architecture shape
        grad_W1 += error_signal * (X1 * Y1)
        grad_B1 += error_signal * Y1
        grad_W2 += error_signal * R * (X2 * Y2)
        grad_B2 += error_signal * R * Y2
        grad_intercept += error_signal  # Intercept derivative is always just 1 * error_signal
        
    # 2. REFLECTION STEP (Average the changes and update parameters)
    # Divide by 11 to get the average direction across your whole dataset
    num_nets = len(meta_dataset)
    
    W[0] -= alpha * (grad_W1 / num_nets)
    B[0] -= alpha * (grad_B1 / num_nets)
    W[1] -= alpha * (grad_W2 / num_nets)
    B[1] -= alpha * (grad_B2 / num_nets)
    global_intercept -= alpha * (grad_intercept / num_nets)
    
    # Track optimization progress
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Total Batch Error: {total_loss:.4f} | Intercept: {global_intercept:.4f}")