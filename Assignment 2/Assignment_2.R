library(keras)
library(ggplot2)

imb_dir<-"./aclImdb"

train_dir<-file.path(imdb_dir, "train")

labels<-c()
texts<-c()

for (label_type in c("neg", "pos")) {
  label<-switch(label_type, neg=0, pos=1)
  dir_name<-file.path(train_dir, label_type)
  for (fname in list.files(dir_name, pattern=glob2rx("*.txt"), full.names=TRUE)){
    texts<-c(texts, readChar(fname, file.info(fname)$size))
    labels<-c(labels, label)
  }
}

training_samples<-200

validation_samples<-10000

max_features <- 10000  # Number of words to consider as features

maxlen <- 150          # Cut texts after this number of words 

max_words<-10000

tokenizer<-text_tokenizer(num_words=max_words)%>% fit_text_tokenizer(texts)

sequences<-texts_to_sequences(tokenizer, texts)

word_index=tokenizer$word_index

labels<-as.array(labels)

data<-pad_sequences(sequences, maxlen=maxlen)

indices<-sample(1:nrow(data))
training_indices<-indices[1:training_samples]
validation_indices<-indices[(training_samples+1):(training_samples+validation_samples)]

x_train<-data[training_indices, ]
y_train<-labels[training_indices]

x_val<-data[validation_indices,]
y_val<-labels[validation_indices]

# (among top max_features most common words)

#Embedded Word Index

lines<-readLines("./glove.6B.300d.txt")

embeddings_index<-new.env(hash=TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
  line<-lines[[i]]
  values<-strsplit(line, " ")[[1]]
  word<-values[[1]]
  embeddings_index[[word]]<-as.double(values[-1])
}

cat("Found", length(embeddings_index), "word vectors. \n")

embedding_dim<-100

embedding_matrix<-array(0, c(max_words, embedding_dim))

for (word in names(word_index)) {
  index<- word_index[[word]]
  if (index<max_words) {
    embedding_vector<embeddings_index[[word]]
    if (!is.null(embedding_vector))
      embedding_matrix[index+1, ]<- embedding_vector
  }
  
}

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


get_layer(model, index=1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

model %>% compile(
  
  optimizer = "rmsprop",
  
  loss = "binary_crossentropy",
  
  metrics = c("acc")
  
)

#The data that was used for validation was created using a validation split of 40%. 

history <- model %>% fit(
  
  x_train, y_train,
  
  epochs = 10,
  
  batch_size = 100,
  
  validation_split = 0.4
  
)

#The Visual Shows a clear improvement in validation accuracy over time with the model used.
  
plot(history)

history %>% save_model_hdf5("Assignment_2.h5")
