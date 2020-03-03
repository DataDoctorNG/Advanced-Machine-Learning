library(keras)
library(ggplot2)

max_features <- 10000  # Number of words to consider as features

maxlen <- 150          # Cut texts after this number of words 

# (among top max_features most common words)

#Embedded Word Index

# Load data

imdb <- dataset_imdb(num_words = max_features)

c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb



# Reverse sequences

x_train <- lapply(x_train, rev) 

x_test <- lapply(x_test, rev) 



# Pad sequences

x_train <- pad_sequences(x_train, maxlen = maxlen)

x_test <- pad_sequences(x_test, maxlen = maxlen)

#After using both and embedding layer and a pretrained word embedding the embedding layer added the most value
#to the model in terms of validation accuracy.

model <- keras_model_sequential() %>% 
  
  layer_embedding(input_dim = max_features, output_dim = 100) %>% 
  
  layer_lstm(units = 32) %>% 
  
  layer_dense(units = 1, activation = "sigmoid")


model %>% compile(
  
  optimizer = "rmsprop",
  
  loss = "binary_crossentropy",
  
  metrics = c("acc")
  
)

#The data that was used for validation was created using a validation split of 40%. 

history <- model %>% fit(
  
  x_train, y_train,
  
  epochs = 10,
  
  batch_size = 128,
  
  validation_split = 0.4
  
)

#The Visual Shows a clear improvement in validation accuracy over time with the model used.
  
plot(history)
