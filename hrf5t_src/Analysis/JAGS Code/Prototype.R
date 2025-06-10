# JAGS implementation of a low parameter version of the prototype model
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
      
      # The similarity of stimulus i to prototype j
      numerator[i,j] = exp(-c*d[i,j])
    }
    # Denominator is just the sum of the numerator values
    denominator[i] = sum(numerator[i,])
  }
  
  # Priors on parameters c, gamma
  c ~ dunif(0,5)

}