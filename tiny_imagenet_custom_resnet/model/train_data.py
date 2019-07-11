# Iterate over epochs.
for epoch in range(3):
    print(f'Epoch {epoch+1}')

  # Iterate over the batches of the dataset.
  for step, x_batch_train in enumerate(train_data):
    with tf.GradientTape() as tape:
      reconstructed = autoencoder(x_batch_train)
      # Compute reconstruction loss
      loss = mse_loss(x_batch_train, reconstructed)
      #loss += sum(autoencoder.losses) 

    grads = tape.gradient(loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))

    loss_metric(loss)

    if step % 100 == 0:
      print(f'Step {step}: mean loss = {loss_metric.result()}')