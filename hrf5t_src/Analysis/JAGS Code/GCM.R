# JAGS implementation of a low parameter version of the GCM
# TODO: Figure out structure for indexing category membership of each stimulus

model{
  # Categorization data
  for (i in 1:nstim){
    # Stimuli categorized according to a multinomial distribution
    y[i,] ~ dmulti(r[i,], t[i])
    predy[i,1:ncat] ~ dmulti(r[i,], t[i])
  }
  
  # Determine the probability values r given to the multinomial distribution
  for (i in 1:nstim){
    for (j in 1:ncat){
      r[i,j] = numerator[i,j]/denominator[i]
      
      # The summed similarity of stimulus i to training stimuli in that category, raised to gamma
      numerator[i,j] = sum(s[i,(ntrainpercat*(j-1)+1):(ntrainpercat*(j-1)+ntrainpercat)])^gamma
    }
    # Denominator is just the sum of the numerator values
    denominator[i] = sum(numerator[i,])
    for (k in 1:ntrain){
      s[i,k] = exp(-c*d[i,train[k]])
    }
  }

  # Priors on parameters c, gamma
  gamma ~ dunif(0,5)
  # gamma <- 2
  c ~ dunif(0,5)
  # c <- 2

}